"""
Vision-Language Retrieval: DAGA vs other PEFT methods
Using COCO Captions dataset with CLIP text encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import json
from PIL import Image
import os

from core.backbones import load_dinov3_backbone
from core.utils import setup_environment
from core.daga import DAGA

# Import other PEFT methods
from paper_experiments.methods.lora import LoRA
from paper_experiments.methods.vpt import VPTDeep
from paper_experiments.methods.adapter_former import AdapterFormer
from paper_experiments.methods.vit_adapter import ViTAdapter

warnings.filterwarnings("ignore")


class COCOCaptionsDataset(Dataset):
    """COCO Captions dataset for vision-language retrieval"""
    
    def __init__(self, data_path, split='val', transform=None, max_samples=5000):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        
        # Load annotations
        ann_file = self.data_path / 'annotations' / f'captions_{split}2017.json'
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Build image id to filename mapping
        self.id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        # Group captions by image
        self.image_to_captions = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_captions:
                self.image_to_captions[img_id] = []
            self.image_to_captions[img_id].append(ann['caption'])
        
        # Get unique image ids (limit for efficiency)
        self.image_ids = list(self.image_to_captions.keys())[:max_samples]
        
        print(f"✓ Loaded {len(self.image_ids)} images with captions")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        filename = self.id_to_filename[img_id]
        img_path = self.data_path / f'{self.split}2017' / filename
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Return first caption for simplicity (5 captions per image)
        captions = self.image_to_captions[img_id]
        
        return image, captions, idx


class VLRetrievalModel(nn.Module):
    """Vision-Language Retrieval model with PEFT support"""
    
    def __init__(
        self,
        pretrained_vit,
        method='baseline',
        adaptation_layers=[1, 2, 10, 11],
    ):
        super().__init__()
        self.vit = pretrained_vit
        self.method = method
        self.adaptation_layers = adaptation_layers
        self.feature_dim = self.vit.embed_dim
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Initialize PEFT modules
        if method == 'baseline':
            pass
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
                rank=8
            )
        elif method == 'vpt-deep':
            self.vpt = VPTDeep(
                embed_dim=self.feature_dim,
                num_prompts=10,
                vpt_layers=adaptation_layers
            )
        elif method == 'adapter-former':
            self.adapter = AdapterFormer(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                bottleneck_dim=64
            )
        elif method == 'vit-adapter':
            self.vit_adapter = ViTAdapter(
                embed_dim=self.feature_dim,
                adapter_layers=adaptation_layers,
                mlp_ratio=0.25
            )
        elif method == 'layer-finetuning':
            for idx in adaptation_layers:
                if idx < len(self.vit.blocks):
                    for param in self.vit.blocks[idx].parameters():
                        param.requires_grad = True
        elif method == 'full-finetune':
            for param in self.vit.parameters():
                param.requires_grad = True
        
        # Projection head to match CLIP embedding dimension (512)
        self.proj = nn.Linear(self.feature_dim, 512)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"✓ VLRetrievalModel ({method}): {trainable:,} / {total:,} trainable params")
    
    def forward(self, x):
        """Extract image features"""
        B = x.shape[0]
        x, (H, W) = self.vit.prepare_tokens_with_masks(x)
        num_patches = H * W
        num_registers = self.vit.num_registers if hasattr(self.vit, 'num_registers') else 0
        
        guidance_map = None
        
        for idx, block in enumerate(self.vit.blocks):
            rope_sincos = self.vit.rope_embed(H=H, W=W) if self.vit.rope_embed else None
            
            # Apply VPT
            if self.method == 'vpt-deep':
                x, _ = self.vpt.insert_prompts(x, idx)
            
            x = block(x, rope_sincos)
            
            # Apply other adaptations
            if self.method == 'daga' and idx in self.adaptation_layers:
                if guidance_map is None:
                    guidance_map = torch.ones(B, H, W, device=x.device) / num_patches
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
        
        # Use CLS token
        features = x[:, 0]
        
        # Project to CLIP space
        features = self.proj(features)
        features = F.normalize(features, p=2, dim=1)
        
        return features


def get_clip_text_encoder(device):
    """Load CLIP text encoder"""
    import open_clip
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def encode_texts(clip_model, tokenizer, texts, device, batch_size=256):
    """Encode texts using CLIP"""
    all_features = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts).to(device)
        
        with torch.no_grad():
            text_features = clip_model.encode_text(tokens)
            text_features = F.normalize(text_features, p=2, dim=1)
        
        all_features.append(text_features.cpu())
    
    return torch.cat(all_features, dim=0)


def extract_image_features(model, dataloader, device):
    """Extract image features"""
    model.eval()
    all_features = []
    all_captions = []
    
    with torch.no_grad():
        for imgs, captions, _ in tqdm(dataloader, desc="Extracting images"):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_features.append(feats.cpu())
            all_captions.extend(captions)
    
    return torch.cat(all_features, dim=0), all_captions


def compute_retrieval_metrics(image_features, text_features, k_vals=[1, 5, 10]):
    """Compute I2T and T2I retrieval metrics"""
    # Similarity matrix: (num_images, num_texts)
    sim = image_features @ text_features.T
    
    num_images = image_features.shape[0]
    
    # For COCO, each image has 5 captions
    # Ground truth: image i matches texts [5*i, 5*i+1, ..., 5*i+4]
    
    results = {}
    
    # Image-to-Text (I2T)
    i2t_recalls = {k: 0.0 for k in k_vals}
    for i in range(num_images):
        gt_texts = list(range(5*i, 5*i+5))
        sorted_indices = torch.argsort(sim[i], descending=True).numpy()
        
        for k in k_vals:
            if any(idx in gt_texts for idx in sorted_indices[:k]):
                i2t_recalls[k] += 1
    
    for k in k_vals:
        i2t_recalls[k] = i2t_recalls[k] / num_images * 100
    
    results['I2T'] = i2t_recalls
    
    # Text-to-Image (T2I)
    t2i_recalls = {k: 0.0 for k in k_vals}
    num_texts = text_features.shape[0]
    
    for t in range(num_texts):
        gt_image = t // 5  # Each 5 texts belong to 1 image
        sorted_indices = torch.argsort(sim[:, t], descending=True).numpy()
        
        for k in k_vals:
            if gt_image in sorted_indices[:k]:
                t2i_recalls[k] += 1
    
    for k in k_vals:
        t2i_recalls[k] = t2i_recalls[k] / num_texts * 100
    
    results['T2I'] = t2i_recalls
    
    return results


def get_transforms(input_size=224):
    """Get image transforms"""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def contrastive_loss(image_features, text_features, temperature=0.07):
    """InfoNCE contrastive loss for image-text alignment"""
    # image_features: (B, D), text_features: (B, D)
    logits = image_features @ text_features.T / temperature  # (B, B)
    labels = torch.arange(len(image_features), device=image_features.device)
    
    # Symmetric loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def train_alignment(model, clip_model, tokenizer, dataloader, device, 
                    epochs=3, lr=1e-4, temperature=0.07):
    """Train image-text alignment with contrastive learning"""
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=lr, weight_decay=0.01
    )
    
    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    print(f"\nTraining alignment for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, captions, _ in pbar:
            imgs = imgs.to(device)
            
            # Get image features
            image_features = model(imgs)
            
            # Get text features (use first caption per image)
            first_captions = [caps[0] for caps in captions]
            tokens = tokenizer(first_captions).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(tokens)
                text_features = F.normalize(text_features, p=2, dim=1)
            
            # Compute contrastive loss
            loss = contrastive_loss(image_features, text_features, temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch {epoch+1}: avg_loss = {avg_loss:.4f}")
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['baseline', 'daga', 'layer-finetuning', 
                        'full-finetune', 'lora', 'vpt-deep', 'adapter-former', 
                        'vit-adapter'], default='baseline')
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="dinov3_vitb16")
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--adaptation_layers", type=int, nargs="+", default=[1, 2, 10, 11])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--max_samples", type=int, default=5000, 
                        help="Max images for evaluation (5000 = COCO 5K test)")
    parser.add_argument("--output_dir", type=str, default="outputs/vl_retrieval")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--train_samples", type=int, default=10000, 
                        help="Samples for training (from train split)")
    args = parser.parse_args()
    
    setup_environment(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Vision-Language Retrieval: {args.method}")
    print(f"{'='*60}")
    
    # Load dataset
    transform = get_transforms(args.input_size)
    dataset = COCOCaptionsDataset(
        args.data_path, 
        split='val', 
        transform=transform,
        max_samples=args.max_samples
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]),
            [item[1] for item in x],
            [item[2] for item in x]
        )
    )
    
    # Load CLIP text encoder
    print("\nLoading CLIP text encoder...")
    clip_model, tokenizer = get_clip_text_encoder(device)
    
    # Load vision model
    print("\nLoading vision model...")
    vit = load_dinov3_backbone(args.model_name, args.pretrained_path)
    model = VLRetrievalModel(
        vit,
        method=args.method,
        adaptation_layers=args.adaptation_layers
    ).to(device)
    
    # Training phase: align DINOv3 features to CLIP text space
    if args.epochs > 0:
        print(f"\n{'='*60}")
        print(f"Training Phase: Aligning DINOv3 to CLIP space")
        print(f"{'='*60}")
        
        # Load training data
        train_dataset = COCOCaptionsDataset(
            args.data_path, 
            split='train', 
            transform=transform,
            max_samples=args.train_samples
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda x: (
                torch.stack([item[0] for item in x]),
                [item[1] for item in x],
                [item[2] for item in x]
            )
        )
        
        # Train alignment
        model = train_alignment(
            model, clip_model, tokenizer, train_loader, device,
            epochs=args.epochs, lr=args.lr
        )
        
        # Save trained model
        ckpt_path = output_dir / f"{args.method}_aligned.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"✓ Saved aligned model to {ckpt_path}")
    
    # Evaluation phase
    print(f"\n{'='*60}")
    print(f"Evaluation Phase")
    print(f"{'='*60}")
    
    # Extract image features
    print("\nExtracting image features...")
    image_features, all_captions = extract_image_features(model, dataloader, device)
    
    # Flatten captions (5 per image)
    flat_captions = []
    for caps in all_captions:
        flat_captions.extend(caps)
    
    # Extract text features
    print(f"\nEncoding {len(flat_captions)} captions...")
    text_features = encode_texts(clip_model, tokenizer, flat_captions, device)
    
    # Compute metrics
    print("\nComputing retrieval metrics...")
    results = compute_retrieval_metrics(image_features, text_features)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results ({args.method})")
    print(f"{'='*60}")
    print(f"\nImage-to-Text Retrieval:")
    for k, v in results['I2T'].items():
        print(f"  R@{k}: {v:.2f}%")
    
    print(f"\nText-to-Image Retrieval:")
    for k, v in results['T2I'].items():
        print(f"  R@{k}: {v:.2f}%")
    
    # Save results
    results_file = output_dir / f"{args.method}_coco_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Method: {args.method}\n")
        f.write(f"Dataset: COCO Captions (val, {args.max_samples} images)\n\n")
        f.write("Image-to-Text:\n")
        for k, v in results['I2T'].items():
            f.write(f"  R@{k}: {v:.2f}%\n")
        f.write("\nText-to-Image:\n")
        for k, v in results['T2I'].items():
            f.write(f"  R@{k}: {v:.2f}%\n")
    
    print(f"\n✅ Results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()

