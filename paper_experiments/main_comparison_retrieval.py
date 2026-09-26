"""
Retrieval comparison: DAGA vs Layer-Finetuning vs Baseline
Simplified version for fair comparison
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import os

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment
from core.datasets.retrieval_datasets import get_retrieval_dataset
from core.daga import DAGA

# Import other PEFT methods
from paper_experiments.methods.lora import LoRA
from paper_experiments.methods.vpt import VPTDeep
from paper_experiments.methods.adapter_former import AdapterFormer
from paper_experiments.methods.vit_adapter import ViTAdapter

warnings.filterwarnings("ignore")


class RetrievalComparisonModel(nn.Module):
    """Retrieval model with PEFT method support"""
    
    def __init__(
        self,
        pretrained_vit,
        method='baseline',
        adaptation_layers=[9, 10, 11],
        pooling='gem',
        gem_p=3.0,
        lora_rank=8,
        vpt_num_prompts=10,
        adapter_bottleneck=64,
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.method = method
        self.adaptation_layers = adaptation_layers
        self.feature_dim = self.vit.embed_dim
        self.pooling = pooling
        self.gem_p = nn.Parameter(torch.ones(1) * gem_p)
        
        # Freeze backbone by default
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize method-specific modules
        if method == 'baseline':
            pass  # No trainable params
            
        elif method == 'daga':
            self.daga = DAGA(
                feature_dim=self.feature_dim,
                daga_layers=adaptation_layers,
                guidance_dim=128,
                mlp_ratio=0.25,
                drop_rate=0.1,
            )
            
        elif method == 'lora':
            self.lora = LoRA(
                embed_dim=self.feature_dim,
                lora_layers=adaptation_layers,
                rank=lora_rank
            )
            
        elif method == 'vpt-deep':
            self.vpt = VPTDeep(
                embed_dim=self.feature_dim,
                num_prompts=vpt_num_prompts,
                vpt_layers=adaptation_layers
            )
            
        elif method == 'adapter-former':
            self.adapter = AdapterFormer(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                bottleneck_dim=adapter_bottleneck
            )
            
        elif method == 'vit-adapter':
            self.vit_adapter = ViTAdapter(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                mlp_ratio=0.25
            )
            
        elif method == 'layer-finetuning':
            # Unfreeze specified layers
            for idx in adaptation_layers:
                if idx < len(self.vit.blocks):
                    for param in self.vit.blocks[idx].parameters():
                        param.requires_grad = True
                        
        elif method == 'full-finetune':
            for param in self.vit.parameters():
                param.requires_grad = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"✓ RetrievalModel ({method}): {trainable:,} / {total:,} trainable params")
    
    def gem_pooling(self, x, p=3.0, eps=1e-6):
        """Generalized Mean Pooling"""
        return (x.clamp(min=eps).pow(p).mean(dim=1)).pow(1.0 / p)
    
    def forward(self, x):
        """Extract features"""
        B = x.shape[0]
        # Use the model's prepare_tokens method
        x, (H, W) = self.vit.prepare_tokens_with_masks(x)
        
        num_patches = H * W
        num_registers = self.vit.num_registers if hasattr(self.vit, 'num_registers') else 0
        
        # Forward through blocks
        guidance_map = None
        num_prompts_inserted = 0
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # 1. Apply VPT prompts insertion
            if self.method == 'vpt-deep':
                x, n_prompts = self.vpt.insert_prompts(x, idx)
                num_prompts_inserted += n_prompts
            
            # 2. Main block forward
            x = block(x, rope_sincos)
            
            # 3. Apply other adaptations
            if self.method == 'daga' and idx in self.adaptation_layers:
                # Create uniform attention map (B, H, W) for DAGA
                if guidance_map is None:
                    guidance_map = torch.ones(B, H, W, device=x.device) / num_patches
                
                # Patch tokens start after CLS (1) and registers
                start_idx = 1 + num_registers
                patch_tokens = x[:, start_idx:start_idx+num_patches]
                adapted = self.daga.apply(patch_tokens, guidance_map, idx, H, W)
                x = torch.cat([x[:, :start_idx], adapted, x[:, start_idx+num_patches:]], dim=1)
                
            elif self.method == 'lora' and idx in self.adaptation_layers:
                x = self.lora(x, idx)
                
            elif self.method == 'adapter-former' and idx in self.adaptation_layers:
                x = self.adapter(x, idx)
                
            elif self.method == 'vit-adapter' and idx in self.adaptation_layers:
                x = self.vit_adapter(x, idx)
        
        x = self.vit.norm(x)
        
        # Pooling
        # Handle shifted indices due to prompts
        # x: [CLS, prompts..., registers..., patches...]
        # So CLS is at 0, patches are at end
        
        if self.pooling == 'cls':
            features = x[:, 0]
        else:
            # Patch tokens are at the end
            patch_tokens = x[:, -num_patches:]
            if self.pooling == 'avg':
                features = patch_tokens.mean(dim=1)
            else:  # gem
                features = self.gem_pooling(patch_tokens, p=self.gem_p)
        
        # L2 normalize
        features = F.normalize(features, p=2, dim=1)
        return features


def extract_features(model, dataset, batch_size, num_workers, device):
    """Extract features from dataset"""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    features = []
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device)
            feats = model(imgs)
            features.append(feats.cpu())
    
    return torch.cat(features, dim=0)


def compute_map(ranks, gnd, kappas=[1, 5, 10]):
    """Compute mAP and recalls"""
    nq = len(gnd)
    aps = np.zeros(nq)
    recalls = {k: np.zeros(nq) for k in kappas}
    
    for i in range(nq):
        qgnd = np.array(gnd[i]['ok'])
        if len(qgnd) == 0:
            continue
            
        qrank = ranks[:, i]
        pos = np.in1d(qrank, qgnd)
        nrel = len(qgnd)
        
        # AP
        rel_rank = np.where(pos)[0] + 1
        if len(rel_rank) > 0:
            precision = np.arange(1, len(rel_rank) + 1) / rel_rank
            aps[i] = np.sum(precision) / nrel
        
        # Recall@k
        for k in kappas:
            recalls[k][i] = np.sum(pos[:k]) / nrel
    
    return np.mean(aps) * 100, {k: np.mean(v) * 100 for k, v in recalls.items()}


def evaluate_retrieval(db_feats, query_feats, retrieval_dataset):
    """Evaluate retrieval performance"""
    # Compute similarity
    sim = query_feats @ db_feats.T
    ranks = torch.argsort(sim, dim=1, descending=True).numpy()
    
    num_queries = retrieval_dataset.get_num_queries()
    
    # Compute metrics for different protocols
    results = {}
    for protocol in ['easy', 'medium', 'hard']:
        gnd_protocol = []
        for q_idx in range(num_queries):
            easy, hard, _ = retrieval_dataset.get_query_ground_truth(q_idx)
            if protocol == 'easy':
                ok = easy
            elif protocol == 'medium':
                ok = easy + hard
            else:
                ok = hard
            gnd_protocol.append({'ok': ok})
        
        mAP, recalls = compute_map(ranks.T, gnd_protocol)
        results[protocol] = {'mAP': mAP, 'recalls': recalls}
    
    return results


def train_epoch(model, db_dataset, query_dataset, optimizer, device):
    """Train one epoch with triplet loss"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    num_queries = query_dataset.retrieval_dataset.get_num_queries()
    
    for q_idx in range(num_queries):
        easy, hard, _ = query_dataset.retrieval_dataset.get_query_ground_truth(q_idx)
        positives = easy + hard
        if len(positives) == 0:
            continue
        
        # Get triplet
        q_img, _ = query_dataset[q_idx]
        pos_idx = np.random.choice(positives)
        pos_img, _ = db_dataset[pos_idx]
        
        neg_indices = list(set(range(len(db_dataset))) - set(positives))
        neg_idx = np.random.choice(neg_indices)
        neg_img, _ = db_dataset[neg_idx]
        
        # Stack and forward
        batch = torch.stack([q_img, pos_img, neg_img]).to(device)
        
        optimizer.zero_grad()
        feats = model(batch)
        q_feat, pos_feat, neg_feat = feats[0], feats[1], feats[2]
        
        # Triplet loss
        pos_dist = (q_feat - pos_feat).pow(2).sum()
        neg_dist = (q_feat - neg_feat).pow(2).sum()
        loss = F.relu(pos_dist - neg_dist + 0.5)
        
        if loss.item() > 0:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['baseline', 'daga', 'layer-finetuning', 
                        'full-finetune', 'lora', 'vpt-deep', 'adapter-former', 
                        'vit-adapter'], default='baseline')
    parser.add_argument("--dataset", choices=['roxford5k', 'rparis6k'], default='roxford5k')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[9, 10, 11])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs/retrieval")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    setup_environment(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Retrieval: {args.method} on {args.dataset}")
    print(f"{'='*60}")
    
    # Load dataset
    db_dataset, query_dataset = get_retrieval_dataset(
        args.dataset, args.data_path, args.input_size
    )
    print(f"✓ Dataset: {len(db_dataset)} db, {len(query_dataset)} queries")
    
    # Load model
    vit = load_dinov3_backbone(args.model_name, args.pretrained_path)
    model = RetrievalComparisonModel(
        vit, 
        method=args.method,
        adaptation_layers=args.adaptation_layers,
        lora_rank=8,
        vpt_num_prompts=10,
        adapter_bottleneck=64,
    ).to(device)
    
    # Training (if not baseline)
    if args.method != 'baseline' and args.epochs > 0:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) > 0:
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
            
            best_mAP = 0
            print(f"\nTraining for {args.epochs} epochs (LR={args.lr})...")
            
            for epoch in range(args.epochs):
                loss = train_epoch(model, db_dataset, query_dataset, optimizer, device)
                
                # Evaluate
                db_feats = extract_features(model, db_dataset, args.batch_size, 
                                           args.num_workers, device)
                query_feats = extract_features(model, query_dataset, args.batch_size,
                                              args.num_workers, device)
                results = evaluate_retrieval(db_feats, query_feats, 
                                            query_dataset.retrieval_dataset)
                
                mAP_m = results['medium']['mAP']
                marker = " ⭐" if mAP_m > best_mAP else ""
                best_mAP = max(best_mAP, mAP_m)
                
                print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f} | "
                      f"mAP(E/M/H): {results['easy']['mAP']:.1f}/"
                      f"{results['medium']['mAP']:.1f}/{results['hard']['mAP']:.1f}{marker}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    
    db_feats = extract_features(model, db_dataset, args.batch_size, 
                               args.num_workers, device)
    query_feats = extract_features(model, query_dataset, args.batch_size,
                                  args.num_workers, device)
    results = evaluate_retrieval(db_feats, query_feats, 
                                query_dataset.retrieval_dataset)
    
    # Print results
    print(f"\n📊 Results ({args.method} on {args.dataset}):")
    for protocol in ['easy', 'medium', 'hard']:
        print(f"  {protocol.upper()}: mAP={results[protocol]['mAP']:.2f}%")
    
    # Save results
    results_file = output_dir / f"{args.method}_{args.dataset}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Method: {args.method}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"LR: {args.lr}\n\n")
        for protocol in ['easy', 'medium', 'hard']:
            f.write(f"{protocol.upper()}: mAP={results[protocol]['mAP']:.2f}%\n")
    
    print(f"\n✅ Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()
