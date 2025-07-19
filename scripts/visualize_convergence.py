#!/usr/bin/env python3
"""
Script to visualize convergence of different NATR training approaches
Creates graphs showing Purchase Recall@20, Item Coverage@20, and Purchase MRR convergence
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_metrics_history(file_path):
    """Load metrics history from JSON or CSV file"""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def create_convergence_plots(metrics_data, output_dir):
    """
    Create convergence plots for multiple models
    
    Args:
        metrics_data: Dict mapping model names to their metrics history
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 4 subplots vertically
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    fig.suptitle('Test Set Performance Convergence Comparison', fontsize=16, y=0.995)
    
    # Define colors and line styles for different models
    colors = {
        'NATR Baseline': '#2E86AB',
        'NATR Oversampled': '#F24236',
        'NATR Negative Sampling': '#F6AE2D',
        'NATR Pre-train + Fine-tune': '#2F4550'
    }
    
    line_styles = {
        'NATR Baseline': '-',
        'NATR Oversampled': '--',
        'NATR Negative Sampling': '-.',
        'NATR Pre-train + Fine-tune': '-'
    }
    
    # Metrics to plot
    metrics = [
        ('purchase_recall@20', 'Purchase Recall@20 (%)', axes[0]),
        ('purchase_recall@10', 'Purchase Recall@10 (%)', axes[1]),
        ('item_coverage@20', 'Item Coverage@20 (%)', axes[2]),
        ('purchase_mrr', 'Purchase MRR', axes[3])
    ]
    
    # Plot each metric
    for metric_key, metric_name, ax in metrics:
        for model_name, history in metrics_data.items():
            if not history:
                continue
                
            epochs = [m.get('epoch', i+1) for i, m in enumerate(history)]
            
            # Handle different metric formats (some might be in percentage, some not)
            if metric_key == 'purchase_mrr':
                values = [m.get(metric_key, 0) for m in history]
            else:
                # Convert to percentage if not already
                values = [m.get(metric_key, 0) * 100 if m.get(metric_key, 0) <= 1 else m.get(metric_key, 0) 
                         for m in history]
            
            # Plot with custom style
            ax.plot(epochs, values, 
                   label=model_name,
                   color=colors.get(model_name, 'black'),
                   linestyle=line_styles.get(model_name, '-'),
                   linewidth=2,
                   marker='o' if len(epochs) < 20 else None,
                   markersize=4,
                   alpha=0.9)
            
            # Add phase transition line for pre-train/fine-tune model
            if 'Pre-train + Fine-tune' in model_name:
                # Find transition point between phases
                pretrain_epochs = [i for i, m in enumerate(history) if m.get('phase') == 'pretrain']
                if pretrain_epochs:
                    transition_epoch = len(pretrain_epochs)
                    ax.axvline(x=transition_epoch, color='gray', linestyle=':', alpha=0.5)
                    # Place text lower to avoid overlap
                    y_position = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15
                    ax.text(transition_epoch, y_position, 'Fine-tune →', 
                           ha='right', va='bottom', fontsize=9, color='gray')
        
        # Customize subplot
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.9)
        
        # Set y-axis limits for better visualization
        if 'Recall' in metric_name or 'Coverage' in metric_name:
            ax.set_ylim(0, max(ax.get_ylim()[1], 100))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as PNG with high DPI for publication
    png_path = os.path.join(output_dir, f'convergence_comparison_{timestamp}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved convergence plot to: {png_path}")
    
    # Save as PDF for LaTeX
    pdf_path = os.path.join(output_dir, f'convergence_comparison_{timestamp}.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"📊 Saved PDF version to: {pdf_path}")
    
    # Also create individual plots for each metric
    for metric_key, metric_name, _ in metrics:
        fig_single, ax_single = plt.subplots(1, 1, figsize=(8, 6))
        
        for model_name, history in metrics_data.items():
            if not history:
                continue
                
            epochs = [m.get('epoch', i+1) for i, m in enumerate(history)]
            
            if metric_key == 'purchase_mrr':
                values = [m.get(metric_key, 0) for m in history]
            else:
                values = [m.get(metric_key, 0) * 100 if m.get(metric_key, 0) <= 1 else m.get(metric_key, 0) 
                         for m in history]
            
            ax_single.plot(epochs, values,
                          label=model_name,
                          color=colors.get(model_name, 'black'),
                          linestyle=line_styles.get(model_name, '-'),
                          linewidth=2.5,
                          marker='o' if len(epochs) < 20 else None,
                          markersize=5,
                          alpha=0.9)
        
        ax_single.set_xlabel('Epoch', fontsize=14)
        ax_single.set_ylabel(metric_name, fontsize=14)
        ax_single.set_title(f'{metric_name} Convergence', fontsize=16)
        ax_single.grid(True, alpha=0.3)
        ax_single.legend(loc='best', framealpha=0.9, fontsize=11)
        
        if 'Recall' in metric_name or 'Coverage' in metric_name:
            ax_single.set_ylim(0, max(ax_single.get_ylim()[1], 100))
        
        plt.tight_layout()
        
        # Save individual metric plot
        metric_filename = metric_key.replace('@', '_at_').replace(' ', '_')
        single_path = os.path.join(output_dir, f'{metric_filename}_convergence_{timestamp}.png')
        plt.savefig(single_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"📊 Saved {metric_name} plot to: {single_path}")
    
    plt.close(fig)

def create_metrics_summary_table(metrics_data, output_dir):
    """Create a summary table of final metrics for all models"""
    summary_data = []
    
    for model_name, history in metrics_data.items():
        if not history:
            continue
            
        # Get final epoch metrics
        final_metrics = history[-1]
        
        # Get best metrics
        best_recall = max([m.get('purchase_recall@20', 0) for m in history])
        best_mrr = max([m.get('purchase_mrr', 0) for m in history])
        best_coverage = max([m.get('item_coverage@20', 0) for m in history])
        
        summary_data.append({
            'Model': model_name,
            'Final Epoch': final_metrics.get('epoch', len(history)),
            'Final Recall@20 (%)': final_metrics.get('purchase_recall@20', 0) * 100 
                                   if final_metrics.get('purchase_recall@20', 0) <= 1 
                                   else final_metrics.get('purchase_recall@20', 0),
            'Best Recall@20 (%)': best_recall * 100 if best_recall <= 1 else best_recall,
            'Final MRR': final_metrics.get('purchase_mrr', 0),
            'Best MRR': best_mrr,
            'Final Coverage@20 (%)': final_metrics.get('item_coverage@20', 0) * 100 
                                     if final_metrics.get('item_coverage@20', 0) <= 1 
                                     else final_metrics.get('item_coverage@20', 0),
            'Best Coverage@20 (%)': best_coverage * 100 if best_coverage <= 1 else best_coverage
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'metrics_summary_table.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n📊 Saved metrics summary table to: {csv_path}")
    
    # Print formatted table
    print("\n📊 Metrics Summary Table:")
    print(df.to_string(index=False, float_format='%.4f'))

def main():
    """Main function to create convergence visualizations"""
    
    # Define paths to metrics files
    graphs_data_dir = 'output/graphs_data'
    
    # Check if directory exists
    if not os.path.exists(graphs_data_dir):
        print(f"⚠️  Creating directory: {graphs_data_dir}")
        os.makedirs(graphs_data_dir)
        print("⚠️  No metrics files found yet. Please run the training scripts first:")
        print("   - train_natr.py (with and without --oversample)")
        print("   - train_natr_sampled_softmax.py")
        print("   - train_natr_enhanced_pre_finetune_fixed.py")
        return
    
    # Load available metrics files
    metrics_data = {}
    
    # Expected files and their model names
    expected_files = {
        'natr_baseline_metrics_history2.json': 'NATR Baseline',
        'natr_oversampled_metrics_history2.json': 'NATR Oversampled',
        'natr_sampled_softmax_metrics_history2.json': 'NATR Negative Sampling',
        'natr_enhanced_pretrain_finetune_metrics_history2.json': 'NATR Pre-train + Fine-tune'
    }
    
    # Also check for CSV files from train_natr_enhanced_pre_finetune_fixed.py
    enhanced_csv_paths = [
        'output/model_info/training_metrics_history.csv',
        'output/model_info_2months/training_metrics_history.csv'
    ]
    
    # Load JSON metrics files first
    for filename, model_name in expected_files.items():
        file_path = os.path.join(graphs_data_dir, filename)
        if os.path.exists(file_path):
            print(f"✅ Loading metrics for {model_name} from {filename}")
            metrics_data[model_name] = load_metrics_history(file_path)
        else:
            print(f"⚠️  Metrics file not found: {file_path}")
    
    # If enhanced model not found, try CSV files as fallback
    if 'NATR Pre-train + Fine-tune' not in metrics_data:
        for csv_path in enhanced_csv_paths:
            if os.path.exists(csv_path):
                print(f"✅ Found enhanced metrics at: {csv_path}")
                metrics_data['NATR Pre-train + Fine-tune'] = load_metrics_history(csv_path)
                break
    
    if not metrics_data:
        print("\n❌ No metrics data found! Please run the training scripts first.")
        return
    
    print(f"\n📊 Found metrics for {len(metrics_data)} models")
    
    # Create visualizations
    create_convergence_plots(metrics_data, graphs_data_dir)
    
    # Create summary table
    create_metrics_summary_table(metrics_data, graphs_data_dir)
    
    print("\n✅ Visualization complete! Check output/graphs_data/ for results.")

if __name__ == "__main__":
    main()