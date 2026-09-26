"""
Quantitative Analysis for DAGA Attention

This script performs quantitative analysis on attention maps,
comparing Baseline vs DAGA models.

Metrics computed:
- Entropy: Measures attention distribution (lower = more focused)
- Gini Coefficient: Measures inequality/concentration (higher = more focused)
- L2 Distance: Magnitude of attention change between models
- Cosine Similarity: Direction similarity between attention patterns

Outputs:
- CSV file with per-sample, per-layer statistics
- Bar plots, heatmaps, box plots, and radar charts
"""

import os
os.environ['SWANLAB_DISABLED'] = '1'

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dinov3"))


# ============================================================================
# Attention Statistics Functions
# ============================================================================

def compute_entropy(attention_map):
    """Compute entropy of attention distribution (lower = more focused)"""
    flat = attention_map.flatten()
    prob = flat / (flat.sum() + 1e-8)
    entropy = -np.sum(prob * np.log(prob + 1e-8))
    return entropy


def compute_gini(attention_map):
    """Compute Gini coefficient (higher = more concentrated/unequal)"""
    sorted_arr = np.sort(attention_map.flatten())
    n = len(sorted_arr)
    cumsum = np.cumsum(sorted_arr)
    return (2 * np.sum((np.arange(1, n+1) * sorted_arr)) / (n * np.sum(sorted_arr) + 1e-8)) - (n + 1) / n


def compute_attention_statistics(baseline_attn, daga_attn):
    """
    Compute all attention statistics comparing baseline vs DAGA
    
    Args:
        baseline_attn: Baseline attention map (H, W)
        daga_attn: DAGA attention map (H, W)
    
    Returns:
        dict with all computed metrics
    """
    # Flatten for some computations
    base_flat = baseline_attn.flatten()
    daga_flat = daga_attn.flatten()
    
    # 1. Entropy
    base_entropy = compute_entropy(baseline_attn)
    daga_entropy = compute_entropy(daga_attn)
    
    # 2. Peak value
    base_peak = baseline_attn.max()
    daga_peak = daga_attn.max()
    
    # 3. Gini coefficient
    base_gini = compute_gini(baseline_attn)
    daga_gini = compute_gini(daga_attn)
    
    # 4. L2 distance
    l2_dist = np.sqrt(np.sum((baseline_attn - daga_attn) ** 2))
    
    # 5. Cosine similarity
    cos_sim = np.dot(base_flat, daga_flat) / (np.linalg.norm(base_flat) * np.linalg.norm(daga_flat) + 1e-8)
    
    # 6. KL divergence (from baseline to DAGA)
    base_prob = base_flat / (base_flat.sum() + 1e-8)
    daga_prob = daga_flat / (daga_flat.sum() + 1e-8)
    kl_div = np.sum(base_prob * np.log((base_prob + 1e-8) / (daga_prob + 1e-8)))
    
    return {
        'baseline_entropy': base_entropy,
        'daga_entropy': daga_entropy,
        'entropy_change': daga_entropy - base_entropy,
        'baseline_peak': base_peak,
        'daga_peak': daga_peak,
        'peak_change': daga_peak - base_peak,
        'baseline_gini': base_gini,
        'daga_gini': daga_gini,
        'gini_change': daga_gini - base_gini,
        'l2_distance': l2_dist,
        'cosine_similarity': cos_sim,
        'kl_divergence': kl_div
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def create_barplot(df, target_layers, output_dir):
    """Create bar plot of average metrics by layer"""
    
    metrics = {
        'entropy_change': 'Entropy Change',
        'gini_change': 'Gini Coefficient Change',
        'l2_distance': 'L2 Distance',
        'cosine_similarity': 'Cosine Similarity',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    layer_labels = [f"Layer {l+1}" for l in target_layers]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[idx]
        
        # Calculate mean and std by layer
        layer_means = []
        layer_stds = []
        for layer in target_layers:
            layer_data = df[df['layer'] == layer][metric_key]
            layer_means.append(layer_data.mean())
            layer_stds.append(layer_data.std())
        
        # Create bar chart
        bars = ax.bar(layer_labels, layer_means, yerr=layer_stds, capsize=5,
                     color=colors[:len(target_layers)],
                     edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Add value labels
        for bar, mean in zip(bars, layer_means):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Layer', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('DAGA vs Baseline: Attention Metrics by Layer', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_layer_barplot.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: metrics_by_layer_barplot.png")


def create_heatmaps(df, target_layers, output_dir):
    """Create heatmap visualizations for each metric"""
    
    metrics = {
        'entropy_change': ('Entropy Change', 'RdYlGn_r'),
        'gini_change': ('Gini Coefficient Change', 'RdYlGn'),
        'l2_distance': ('L2 Distance', 'YlOrRd'),
        'cosine_similarity': ('Cosine Similarity', 'RdYlGn'),
    }
    
    unique_samples = df['sample_idx'].unique()
    
    if len(unique_samples) <= 1:
        print("  ⚠ Not enough samples for heatmap")
        return
    
    for metric_key, (metric_name, cmap) in metrics.items():
        # Create matrix
        matrix = np.zeros((len(unique_samples), len(target_layers)))
        sample_labels = []
        
        for i, sample_idx in enumerate(unique_samples):
            sample_data = df[df['sample_idx'] == sample_idx]
            label = sample_data.iloc[0]['true_class'] if 'true_class' in sample_data.columns else str(sample_idx)
            sample_labels.append(label[:12])  # Truncate
            for j, layer in enumerate(target_layers):
                layer_data = sample_data[sample_data['layer'] == layer]
                if len(layer_data) > 0:
                    matrix[i, j] = layer_data[metric_key].values[0]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, max(8, len(unique_samples) * 0.4)))
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap)
        
        ax.set_xticks(np.arange(len(target_layers)))
        ax.set_xticklabels([f'L{l+1}' for l in target_layers], fontsize=11)
        ax.set_yticks(np.arange(len(unique_samples)))
        ax.set_yticklabels(sample_labels, fontsize=9)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(metric_name, fontsize=11)
        
        # Add value annotations
        for i in range(len(unique_samples)):
            for j in range(len(target_layers)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                              ha='center', va='center', fontsize=8,
                              color='white' if abs(matrix[i, j]) > abs(matrix).max()/2 else 'black')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Sample', fontsize=12)
        ax.set_title(f'{metric_name}\n(Sample × Layer Heatmap)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{metric_key}.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  ✓ Saved: heatmap_{metric_key}.png")


def create_boxplot(df, target_layers, output_dir):
    """Create box plot of metric distributions by layer"""
    
    metrics = {
        'entropy_change': 'Entropy Change',
        'gini_change': 'Gini Coefficient Change',
        'l2_distance': 'L2 Distance',
        'cosine_similarity': 'Cosine Similarity',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[idx]
        
        box_data = [df[df['layer'] == layer][metric_key].values for layer in target_layers]
        
        bp = ax.boxplot(box_data, labels=[f'L{l+1}' for l in target_layers],
                       patch_artist=True, notch=True)
        
        for patch, color in zip(bp['boxes'], colors[:len(target_layers)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_title(f'{metric_name} Distribution', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('DAGA vs Baseline: Metric Distributions by Layer', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_distribution_boxplot.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: metrics_distribution_boxplot.png")


def create_radar_chart(df, target_layers, output_dir):
    """Create radar chart comparing layers"""
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    metric_names = ['Entropy Δ', 'Gini Δ', 'L2 Dist', 'Cos Sim']
    num_metrics = len(metric_names)
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    for i, layer in enumerate(target_layers):
        layer_data = df[df['layer'] == layer]
        
        if len(layer_data) == 0:
            continue
        
        values = [
            layer_data['entropy_change'].mean(),
            layer_data['gini_change'].mean(),
            layer_data['l2_distance'].mean() / 10,  # Scale
            layer_data['cosine_similarity'].mean()
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Layer {layer+1}',
               color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_title('Attention Metrics Radar Chart by Layer', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_radar_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: metrics_radar_chart.png")


def create_summary_table(df, target_layers, output_dir):
    """Create summary statistics table image"""
    
    metrics = ['entropy_change', 'gini_change', 'l2_distance', 'cosine_similarity']
    metric_names = ['Entropy Δ', 'Gini Δ', 'L2 Distance', 'Cosine Sim']
    
    # Build summary data
    summary_data = []
    for layer in target_layers:
        layer_data = df[df['layer'] == layer]
        row = [f'Layer {layer+1}']
        for metric in metrics:
            mean = layer_data[metric].mean()
            std = layer_data[metric].std()
            row.append(f'{mean:.4f} ± {std:.4f}')
        summary_data.append(row)
    
    # Add overall row
    overall_row = ['Overall']
    for metric in metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        overall_row.append(f'{mean:.4f} ± {std:.4f}')
    summary_data.append(overall_row)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=summary_data,
        colLabels=['Layer'] + metric_names,
        cellLoc='center',
        loc='center',
        colColours=['#3498db'] + ['#ecf0f1'] * len(metrics)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(metrics) + 1):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Quantitative Analysis Summary\n(Mean ± Std)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: summary_table.png")


def run_quantitative_analysis(csv_path, output_dir=None, target_layers=None):
    """
    Run quantitative analysis from existing CSV file
    
    Args:
        csv_path: Path to attention_quantitative_analysis.csv
        output_dir: Output directory for visualizations (default: same as CSV)
        target_layers: List of layer indices to analyze
    """
    csv_path = Path(csv_path)
    
    if output_dir is None:
        output_dir = csv_path.parent / "quantitative_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Quantitative Analysis Visualization")
    print("=" * 70)
    print(f"Input CSV: {csv_path}")
    print(f"Output Dir: {output_dir}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Determine target layers from data if not provided
    if target_layers is None:
        target_layers = sorted(df['layer'].unique().tolist())
    
    print(f"Target layers: {target_layers}")
    
    # Print summary statistics
    print("\n" + "-" * 70)
    print("Summary Statistics")
    print("-" * 70)
    print(f"{'Layer':<10} {'Entropy Δ':>14} {'Gini Δ':>14} {'L2 Dist':>14} {'Cos Sim':>14}")
    print("-" * 70)
    
    for layer in target_layers:
        layer_data = df[df['layer'] == layer]
        if len(layer_data) > 0:
            print(f"Layer {layer+1:<4} "
                  f"{layer_data['entropy_change'].mean():>+14.4f} "
                  f"{layer_data['gini_change'].mean():>+14.4f} "
                  f"{layer_data['l2_distance'].mean():>14.4f} "
                  f"{layer_data['cosine_similarity'].mean():>14.4f}")
    
    print("-" * 70)
    print(f"{'Overall':<10} "
          f"{df['entropy_change'].mean():>+14.4f} "
          f"{df['gini_change'].mean():>+14.4f} "
          f"{df['l2_distance'].mean():>14.4f} "
          f"{df['cosine_similarity'].mean():>14.4f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_barplot(df, target_layers, output_dir)
    create_heatmaps(df, target_layers, output_dir)
    create_boxplot(df, target_layers, output_dir)
    create_radar_chart(df, target_layers, output_dir)
    create_summary_table(df, target_layers, output_dir)
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    
    return df


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Quantitative Analysis for DAGA Attention")
    
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to attention_quantitative_analysis.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for visualizations")
    parser.add_argument("--target_layers", type=int, nargs="+", default=None,
                       help="Specific layers to analyze (default: all from CSV)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_quantitative_analysis(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        target_layers=args.target_layers
    )

