# utils/metrics_visualization.py
"""
Visualization utilities for recommendation metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import os


def plot_training_curves(history: Dict[str, List], save_path: Optional[str] = None):
    """
    Plot training curves for loss and key metrics
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    if 'train_loss' in history and 'test_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(history['test_loss'], label='Test', linewidth=2)
        axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Hit@10
    if 'train_metrics' in history and 'test_metrics' in history:
        train_hit10 = [m.get('hit@10', 0) * 100 for m in history['train_metrics']]
        test_hit10 = [m.get('hit@10', 0) * 100 for m in history['test_metrics']]
        
        axes[0, 1].plot(train_hit10, label='Train', linewidth=2)
        axes[0, 1].plot(test_hit10, label='Test', linewidth=2)
        axes[0, 1].set_title('Hit@10', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Hit@10 (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot all Hit@k
    if 'test_metrics' in history:
        k_values = [1, 5, 10, 20]
        for k in k_values:
            test_hitk = [m.get(f'hit@{k}', 0) * 100 for m in history['test_metrics']]
            axes[1, 0].plot(test_hitk, label=f'Hit@{k}', linewidth=2)
        
        axes[1, 0].set_title('Test Hit@k', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Hit@k (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot all MRR@k
    if 'test_metrics' in history:
        k_values = [1, 5, 10, 20]
        for k in k_values:
            test_mrrk = [m.get(f'mrr@{k}', 0) for m in history['test_metrics']]
            axes[1, 1].plot(test_mrrk, label=f'MRR@{k}', linewidth=2)
        
        axes[1, 1].set_title('Test MRR@k', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MRR@k')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                           save_path: Optional[str] = None):
    """
    Plot comparison of metrics across different models
    
    Args:
        metrics_dict: Dictionary of model_name -> metrics
        save_path: Path to save the plot
    """
    # Prepare data for plotting
    models = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    # Filter to only numeric metrics
    metric_names = [m for m in metric_names if isinstance(metrics_dict[models[0]][m], (int, float))]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.8 / len(metric_names)
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[model][metric] for model in models]
        offset = (i - len(metric_names)/2) * width
        bars = ax.bar(x + offset, values, width, label=metric)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_attention_heatmap(attention_weights: np.ndarray, 
                          labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
    """
    Plot attention weights as a heatmap
    
    Args:
        attention_weights: Attention weights array
        labels: Labels for the items
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(attention_weights, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=labels if labels else True,
                yticklabels=labels if labels else True,
                cbar_kws={'label': 'Attention Weight'})
    
    plt.title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Items')
    plt.ylabel('Samples')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_metrics_report(history: Dict[str, List], 
                         save_dir: str = 'reports'):
    """
    Create a comprehensive metrics report
    
    Args:
        history: Training history
        save_dir: Directory to save the report
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract final metrics
    final_metrics = {}
    if 'test_metrics' in history and len(history['test_metrics']) > 0:
        final_metrics = history['test_metrics'][-1]
    
    # Create summary
    summary = {
        'training_epochs': len(history.get('train_loss', [])),
        'best_epoch': 0,
        'best_hit@10': 0,
        'final_metrics': final_metrics
    }
    
    # Find best epoch
    if 'test_metrics' in history:
        hit10_values = [m.get('hit@10', 0) for m in history['test_metrics']]
        if hit10_values:
            best_idx = np.argmax(hit10_values)
            summary['best_epoch'] = best_idx + 1
            summary['best_hit@10'] = hit10_values[best_idx]
    
    # Save metrics history as CSV
    if 'test_metrics' in history:
        metrics_df = pd.DataFrame(history['test_metrics'])
        metrics_df.to_csv(os.path.join(save_dir, 'test_metrics.csv'), index=False)
    
    # Save summary as JSON
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Create plots
    plot_training_curves(history, os.path.join(save_dir, 'training_curves.png'))
    
    # Create markdown report
    report = f"""# Training Report

## Summary
- **Training Epochs**: {summary['training_epochs']}
- **Best Epoch**: {summary['best_epoch']}
- **Best Hit@10**: {summary['best_hit@10']:.4f}

## Final Metrics
"""
    
    for metric, value in summary['final_metrics'].items():
        report += f"- **{metric}**: {value:.4f}\n"
    
    report += "\n## Training Curves\n![Training Curves](training_curves.png)\n"
    
    with open(os.path.join(save_dir, 'report.md'), 'w') as f:
        f.write(report)
    
    print(f"Report saved to {save_dir}")


# Example usage
if __name__ == "__main__":
    # Example training history
    history = {
        'train_loss': [0.5, 0.4, 0.35, 0.3, 0.28],
        'test_loss': [0.48, 0.42, 0.38, 0.36, 0.35],
        'train_metrics': [
            {'hit@10': 0.1, 'mrr@10': 0.05},
            {'hit@10': 0.15, 'mrr@10': 0.08},
            {'hit@10': 0.2, 'mrr@10': 0.12},
            {'hit@10': 0.22, 'mrr@10': 0.14},
            {'hit@10': 0.24, 'mrr@10': 0.15}
        ],
        'test_metrics': [
            {'hit@10': 0.12, 'hit@1': 0.05, 'hit@5': 0.08, 'hit@20': 0.15,
             'mrr@10': 0.06, 'mrr@1': 0.05, 'mrr@5': 0.055, 'mrr@20': 0.065},
            {'hit@10': 0.16, 'hit@1': 0.07, 'hit@5': 0.11, 'hit@20': 0.19,
             'mrr@10': 0.09, 'mrr@1': 0.07, 'mrr@5': 0.08, 'mrr@20': 0.095},
            {'hit@10': 0.19, 'hit@1': 0.09, 'hit@5': 0.14, 'hit@20': 0.22,
             'mrr@10': 0.12, 'mrr@1': 0.09, 'mrr@5': 0.11, 'mrr@20': 0.125},
            {'hit@10': 0.21, 'hit@1': 0.10, 'hit@5': 0.16, 'hit@20': 0.24,
             'mrr@10': 0.14, 'mrr@1': 0.10, 'mrr@5': 0.13, 'mrr@20': 0.145},
            {'hit@10': 0.22, 'hit@1': 0.11, 'hit@5': 0.17, 'hit@20': 0.25,
             'mrr@10': 0.15, 'mrr@1': 0.11, 'mrr@5': 0.14, 'mrr@20': 0.155}
        ]
    }
    
    # Plot training curves
    plot_training_curves(history)
    
    # Create metrics report
    create_metrics_report(history)