import os
import sys
import pickle
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Modified PackageEncoder that exposes attention weights
class VisualPackageEncoder(torch.nn.Module):
    """Version of PackageEncoder that exposes attention weights for visualization"""
    
    def __init__(self, original_model):
        """Initialize by copying weights from original model"""
        super(VisualPackageEncoder, self).__init__()
        self.original_model = original_model
        
    def forward(self, titles, country_ids, category_ids, theme_ids, user_query=None):
        """Forward pass that returns both representations and attention weights"""
        batch_size = len(titles)
        device = country_ids.device
        
        # Title encoding
        title_embeddings = self.original_model.process_batch_titles(titles).to(device)
        title_outputs, _ = self.original_model.title_lstm(title_embeddings.unsqueeze(1))
        title_representation = title_outputs.squeeze(1)
        
        # Other views encoding
        country_embedded = self.original_model.country_embeddings(country_ids)
        country_representation = self.original_model.destination_mlp(country_embedded)
        
        category_embedded = self.original_model.category_embeddings(category_ids)
        category_representation = self.original_model.category_mlp(category_embedded)
        
        theme_embedded = self.original_model.theme_embeddings(theme_ids)
        theme_representation = self.original_model.theme_mlp(theme_embedded)
        
        # Stack all representations
        view_representations = torch.stack([
            title_representation, 
            country_representation, 
            category_representation, 
            theme_representation
        ], dim=1)  # [batch_size, 4, hidden_dim]
        
        # View-level attention
        if user_query is not None:
            attention_query = user_query.unsqueeze(1)
        else:
            attention_query = self.original_model.view_attention_query.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        
        view_attn_logits = torch.bmm(attention_query, view_representations.transpose(1, 2)).squeeze(1)
        view_attn_weights = torch.softmax(view_attn_logits, dim=1).unsqueeze(1)
        
        # Apply attention weights
        package_representation = torch.bmm(view_attn_weights, view_representations).squeeze(1)
        
        # Extract view attention weights
        view_attention = view_attn_weights.squeeze(1).detach()
        
        return package_representation, {
            'view_attention': view_attention,
        }

def visualize_attention_weights(model_path, data_path, mappings_path, output_dir, num_samples=100):
    """
    Visualize attention weights in the PackageEncoder
    
    Args:
        model_path: Path to the saved PackageEncoder model
        data_path: Path to the package info data file
        mappings_path: Path to the mappings file
        output_dir: Directory to save visualization results
        num_samples: Number of packages to sample for visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original model
    from models.package_encoder import PackageEncoder
    
    # Load mappings
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    package_to_idx = mappings['package_to_idx']
    country_to_idx = mappings['country_to_idx']
    category_to_idx = mappings['category_to_idx']
    theme_to_idx = mappings['theme_to_idx']
    
    # Create reverse mappings
    idx_to_country = {idx: country for country, idx in country_to_idx.items()}
    idx_to_category = {idx: category for category, idx in category_to_idx.items()}
    idx_to_theme = {idx: theme for theme, idx in theme_to_idx.items()}
    
    # Load package info
    with open(data_path, 'rb') as f:
        package_info = pickle.load(f)
    
    # Load model
    print(f"Loading model from {model_path}")
    original_model = PackageEncoder(
        package_vocab_size=len(package_to_idx) + 1,
        country_vocab_size=len(country_to_idx) + 1,
        category_vocab_size=len(category_to_idx) + 1,
        theme_vocab_size=len(theme_to_idx) + 1,
        fasttext_model_path="data/cc.nl.300.bin"
    )
    original_model.load_state_dict(torch.load(model_path))
    original_model.eval()
    
    # Create visual model
    visual_model = VisualPackageEncoder(original_model)
    visual_model.eval()
    
    # Sample packages
    sample_packages = package_info.sample(min(num_samples, len(package_info)))
    
    # Collect attention weights
    view_attentions = []
    country_categories = []
    
    print("Collecting attention weights...")
    for _, package in tqdm(sample_packages.iterrows(), total=len(sample_packages)):
        title = package['title']
        country = package['country']
        category = package['category']
        theme = package['theme']
        
        # Convert to tensors
        country_id = torch.tensor([country_to_idx.get(country, 0)])
        category_id = torch.tensor([category_to_idx.get(category, 0)])
        theme_id = torch.tensor([theme_to_idx.get(theme, 0)])
        
        # Forward pass
        with torch.no_grad():
            _, attentions = visual_model(
                [title], 
                country_id, 
                category_id, 
                theme_id
            )
        
        view_attentions.append(attentions['view_attention'][0].numpy())
        country_categories.append((country, category))
    
    view_attentions = np.array(view_attentions)
    
    # 1. Overall view attention distribution
    view_names = ['Title', 'Country', 'Category', 'Theme']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[view_attentions[:, i] for i in range(4)])
    plt.xticks(range(4), view_names)
    plt.ylabel('Attention Weight')
    plt.title('Distribution of View Attention Weights')
    plt.savefig(os.path.join(output_dir, 'view_attention_boxplot.png'))
    
    # Calculate mean attention per view
    mean_attention = view_attentions.mean(axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.bar(view_names, mean_attention)
    plt.ylabel('Mean Attention Weight')
    plt.title('Mean Attention Weight per View')
    for i, v in enumerate(mean_attention):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.savefig(os.path.join(output_dir, 'view_attention_means.png'))
    
    # 2. Attention by country
    countries = [c for c, _ in country_categories]
    unique_countries = list(set(countries))
    
    if len(unique_countries) > 1:
        country_attentions = {}
        for i, (country, _) in enumerate(country_categories):
            if country not in country_attentions:
                country_attentions[country] = []
            country_attentions[country].append(view_attentions[i])
        
        # Get top 10 countries by frequency
        top_countries = sorted(
            [(c, len(attn)) for c, attn in country_attentions.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        plt.figure(figsize=(14, 8))
        
        for i, (country, _) in enumerate(top_countries):
            country_attn = np.array(country_attentions[country])
            mean_attn = country_attn.mean(axis=0)
            
            plt.subplot(2, 5, i+1)
            plt.bar(view_names, mean_attn)
            plt.title(f'{country} (n={len(country_attn)})')
            plt.ylim(0, 0.7)
            if i % 5 == 0:
                plt.ylabel('Mean Attention')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'view_attention_by_country.png'))
    
    # 3. Attention by category
    categories = [c for _, c in country_categories]
    unique_categories = list(set(categories))
    
    if len(unique_categories) > 1:
        category_attentions = {}
        for i, (_, category) in enumerate(country_categories):
            if category not in category_attentions:
                category_attentions[category] = []
            category_attentions[category].append(view_attentions[i])
        
        plt.figure(figsize=(14, 8))
        
        for i, category in enumerate(unique_categories[:min(10, len(unique_categories))]):
            if category in category_attentions:
                category_attn = np.array(category_attentions[category])
                mean_attn = category_attn.mean(axis=0)
                
                plt.subplot(2, 5, i+1)
                plt.bar(view_names, mean_attn)
                plt.title(f'{category} (n={len(category_attn)})')
                plt.ylim(0, 0.7)
                if i % 5 == 0:
                    plt.ylabel('Mean Attention')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'view_attention_by_category.png'))
    
    # 4. Correlation between views
    plt.figure(figsize=(8, 6))
    corr_matrix = np.corrcoef(view_attentions.T)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', xticklabels=view_names, yticklabels=view_names)
    plt.title('Correlation Between View Attentions')
    plt.savefig(os.path.join(output_dir, 'view_attention_correlation.png'))
    
    # 5. Sample packages with their attention weights
    num_examples = min(10, len(sample_packages))
    examples = sample_packages.iloc[:num_examples]
    
    plt.figure(figsize=(14, num_examples*3))
    
    for i, (_, package) in enumerate(examples.iterrows()):
        attn = view_attentions[i]
        
        plt.subplot(num_examples, 1, i+1)
        plt.bar(view_names, attn)
        plt.title(f"Package: {package['title'][:50]}...")
        plt.ylim(0, 0.7)
        for j, v in enumerate(attn):
            plt.text(j, v + 0.02, f'{v:.3f}', ha='center')
        
        # Add details as text
        details = f"Country: {package['country']}, Category: {package['category']}, Theme: {package['theme']}"
        plt.figtext(0.5, (i+0.8)/num_examples, details, ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_packages_attention.png'))
    
    print(f"Attention weight visualizations saved to {output_dir}")

def analyze_natr_attention(model, test_loader, output_dir, num_samples=50):
    """
    Analyze attention weights from the enhanced NATR model
    
    Args:
        model: Trained NATR model
        test_loader: DataLoader for test samples
        output_dir: Directory to save visualizations
        num_samples: Number of samples to analyze
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    model.enable_debug(True)
    
    # Collect attention data
    attention_data = {
        'package_attention': [],
        'view_attention': [],
        'package_ids': [],
        'event_types': [],
        'is_purchase': [],
        'timestamps': []
    }
    
    print(f"Collecting attention weights from {num_samples} test samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            # Forward pass with debug enabled
            outputs = model(batch)
            
            # Store attention weights if available
            if 'attention_weights' in outputs:
                for module_name, attn_weights in outputs['attention_weights'].items():
                    if 'package_attention' in module_name:
                        # Extract and store package-level attention
                        package_attention = attn_weights.cpu().numpy()
                        attention_data['package_attention'].append(package_attention)
                        
                        # Also store package IDs and other metadata for analysis
                        attention_data['package_ids'].append(batch['short_term']['package_ids'].cpu().numpy())
                        attention_data['event_types'].append(batch['short_term']['event_types'].cpu().numpy())
                        attention_data['is_purchase'].append(batch.get('is_purchase', torch.zeros(batch['user_id'].size(0))).cpu().numpy())
                        
                        if 'timestamps' in batch['short_term']:
                            attention_data['timestamps'].append(batch['short_term']['timestamps'].cpu().numpy())
    
    model.enable_debug(False)
    
    # If no attention data collected, return
    if not attention_data['package_attention']:
        print("No attention weights collected. Check if debug mode is enabled.")
        return
        
    # Save raw attention data
    with open(os.path.join(output_dir, 'attention_data.pkl'), 'wb') as f:
        pickle.dump(attention_data, f)
    
    # Visualizations
    
    # 1. Package attention distribution
    plt.figure(figsize=(10, 6))
    
    # Flatten all attention weights
    all_attention = np.concatenate([a.flatten() for a in attention_data['package_attention']])
    
    # Plot distribution
    sns.histplot(all_attention, bins=30, kde=True)
    plt.title('Distribution of Package-Level Attention Weights')
    plt.xlabel('Attention Weight')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'package_attention_distribution.png'))
    
    # 2. Average attention by position in sequence
    seq_len = attention_data['package_attention'][0].shape[-1]
    position_attention = np.zeros(seq_len)
    position_counts = np.zeros(seq_len)
    
    for attn, pkg_ids in zip(attention_data['package_attention'], attention_data['package_ids']):
        for i in range(seq_len):
            # Only count positions with valid package IDs
            if pkg_ids[0, i] > 0:
                position_attention[i] += attn[0, 0, 0, i]
                position_counts[i] += 1
    
    # Average by dividing by counts
    avg_position_attention = np.divide(
        position_attention, 
        position_counts, 
        out=np.zeros_like(position_attention), 
        where=position_counts>0
    )
    
    # Plot position attention
    plt.figure(figsize=(12, 6))
    plt.bar(range(seq_len), avg_position_attention)
    plt.title('Average Attention by Position in Sequence')
    plt.xlabel('Position (0 = Most Recent)')
    plt.ylabel('Average Attention Weight')
    plt.savefig(os.path.join(output_dir, 'position_attention.png'))
    
    # 3. Attention by event type
    if attention_data['event_types']:
        event_type_attention = {}
        
        for attn, events, pkg_ids in zip(
            attention_data['package_attention'], 
            attention_data['event_types'],
            attention_data['package_ids']
        ):
            for i in range(seq_len):
                # Only count valid packages
                if pkg_ids[0, i] > 0:
                    event_type = events[0, i].item()
                    if event_type not in event_type_attention:
                        event_type_attention[event_type] = []
                    
                    event_type_attention[event_type].append(attn[0, 0, 0, i])
        
        # Map event types to names for the plot
        event_names = {
            0: 'Padding',
            1: 'ViewContent',
            2: 'AddToCart',
            3: 'InitiateCheckout',
            4: 'Purchase'
        }
        
        # Calculate average attention per event type
        event_avg_attention = {}
        event_std_attention = {}
        for event_type, attns in event_type_attention.items():
            if attns:
                event_avg_attention[event_type] = np.mean(attns)
                event_std_attention[event_type] = np.std(attns)
        
        # Skip padding events (0) in the plot
        events_to_plot = [k for k in sorted(event_avg_attention.keys()) if k > 0]
        
        plt.figure(figsize=(10, 6))
        plt.bar(
            [event_names.get(e, f"Event_{e}") for e in events_to_plot],
            [event_avg_attention[e] for e in events_to_plot],
            yerr=[event_std_attention[e] for e in events_to_plot]
        )
        plt.title('Average Attention by Event Type')
        plt.ylabel('Average Attention Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'event_type_attention.png'))
    
    # 4. Recency effect on attention
    if attention_data['timestamps'] and len(attention_data['timestamps']) > 0:
        # Create bins for time deltas
        time_deltas = []
        attn_weights = []
        
        for attn, times, pkg_ids in zip(
            attention_data['package_attention'],
            attention_data['timestamps'],
            attention_data['package_ids']
        ):
            # Get valid timestamps
            valid_times = []
            valid_attns = []
            valid_positions = []
            
            for i in range(seq_len):
                if pkg_ids[0, i] > 0 and times[0, i] > 0:
                    valid_times.append(times[0, i])
                    valid_attns.append(attn[0, 0, 0, i])
                    valid_positions.append(i)
            
            if valid_times:
                # Calculate time deltas from most recent
                most_recent = max(valid_times)
                for t, a in zip(valid_times, valid_attns):
                    # Convert to hours
                    delta_hours = (most_recent - t) / 3600
                    time_deltas.append(delta_hours)
                    attn_weights.append(a)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(time_deltas, attn_weights, alpha=0.5)
        
        # Add trend line
        if time_deltas:
            z = np.polyfit(time_deltas, attn_weights, 1)
            p = np.poly1d(z)
            plt.plot(sorted(time_deltas), p(sorted(time_deltas)), "r--", linewidth=2)
            
        plt.title('Attention Weight vs. Time Recency')
        plt.xlabel('Hours from Most Recent Event')
        plt.ylabel('Attention Weight')
        plt.xscale('log')  # Log scale for better visualization
        plt.savefig(os.path.join(output_dir, 'recency_attention.png'))
        
    # Save a summary report
    summary = {
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples_analyzed": len(attention_data['package_attention']),
        "average_attention": float(np.mean(all_attention)),
        "std_attention": float(np.std(all_attention)),
        "position_attention": avg_position_attention.tolist(),
        "event_type_attention": {event_names.get(k, f"Event_{k}"): v 
                                for k, v in event_avg_attention.items()} if 'event_avg_attention' in locals() else {}
    }
    
    with open(os.path.join(output_dir, 'attention_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Attention analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    # Check for command line arguments for different modes
    if len(sys.argv) > 1 and sys.argv[1] == "package_encoder":
        # Legacy package encoder analysis
        model_path = "output/package_encoder/package_encoder.pt"
        data_path = "data/processed/package_info.pkl"
        mappings_path = "data/processed/package_mappings.pkl"
        output_dir = "output/package_encoder/attention_visualization"
        
        visualize_attention_weights(model_path, data_path, mappings_path, output_dir)
    else:
        # NATR model analysis (default)
        from models.natr import NATR, NATRConfig
        
        # Load model from checkpoint
        checkpoint_path = "checkpoints/natr/best_model.pth"
        output_dir = "output/natr/attention_analysis"
        
        print(f"Loading NATR model from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Create config from saved dict
        config_dict = checkpoint['config']
        config = NATRConfig(
            num_users=config_dict['num_users'],
            num_packages=config_dict['num_packages'],
            num_countries=config_dict['num_countries'],
            num_categories=config_dict['num_categories'],
            num_themes=config_dict['num_themes'],
            title_embedding_dim=config_dict['title_embedding_dim'],
            hidden_dim=config_dict['hidden_dim'],
            embedding_dim=config_dict['embedding_dim'],
            user_embedding_dim=config_dict['user_embedding_dim'],
            dropout=config_dict['dropout']
        )
        
        # Create model
        model = NATR(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create a mini test dataset to analyze attention
        print("Creating mini dataset for attention analysis...")
        
        try:
            # Load the model's dataset class
            from utils.package_processor import PackageProcessor, TravelPackageDataset
            from utils.session_processor import SessionProcessor
            from torch.utils.data import DataLoader, Subset
            
            # Initialize processors with minimal loading
            print("Initializing processors...")
            package_processor = PackageProcessor(
                data_path="data/feed.parquet",
                cache_dir='data/cache',
                load_coordinates=True,
                load_embeddings=True
            )
            
            session_processor = SessionProcessor(
                data_path="data/bookit_events_data_13_months.parquet",
                cache_dir='data/cache'
            )
            
            # Load cached data
            package_processor.load_data()
            package_processor.create_mappings()
            
            # Load mappings without full data
            user_to_idx = {}
            package_to_idx = {}
            event_to_idx = {'ViewContent': 1, 'AddToCart': 2, 'InitiateCheckout': 3, 'Purchase': 4}
            
            # Try to load mappings from cache
            mappings_cache = os.path.join('data/cache', 'user_package_mappings.pkl')
            if os.path.exists(mappings_cache):
                print(f"Loading mappings from cache: {mappings_cache}")
                with open(mappings_cache, 'rb') as f:
                    mappings = pickle.load(f)
                    user_to_idx = mappings.get('user_to_idx', {})
                    package_to_idx = mappings.get('package_to_idx', {})
            else:
                print("No mappings cache found, using processor mappings")
                session_processor.load_data()
                session_processor.create_mappings()
                user_to_idx = session_processor.user_to_idx
                package_to_idx = session_processor.package_to_idx
            
            # Create mini dataset
            print(f"Creating mini dataset...")
            # Get cached test samples if available
            test_samples_path = "data/cache/test_samples.pkl"
            cached_samples_path = "data/cache/enhanced_training_samples.pkl"
            
            if os.path.exists(test_samples_path):
                print(f"Loading test samples from: {test_samples_path}")
                with open(test_samples_path, 'rb') as f:
                    test_samples = pickle.load(f)
            elif os.path.exists(cached_samples_path):
                print(f"Loading samples from: {cached_samples_path}")
                with open(cached_samples_path, 'rb') as f:
                    all_samples = pickle.load(f)
                # Use a small subset for analysis
                test_samples = all_samples[:1000]
            else:
                print("No samples found for attention analysis")
                sys.exit(1)
            
            print(f"Creating dataset with {len(test_samples)} samples...")
            
            # Create a minimal dataset - only purpose is to analyze attention
            try:
                test_dataset = TravelPackageDataset(
                    test_samples,
                    package_processor,
                    user_to_idx,
                    package_to_idx, 
                    event_to_idx,
                    max_short_term=10,
                    max_long_term=20,
                    use_cache=True
                )
                
                # Create a small subset
                subset_indices = list(range(min(100, len(test_dataset))))
                subset_dataset = Subset(test_dataset, subset_indices)
                
                test_loader = DataLoader(
                    subset_dataset,
                    batch_size=1,
                    shuffle=False
                )
                
                print(f"Analyzing attention weights from {len(subset_indices)} samples...")
                analyze_natr_attention(model, test_loader, output_dir, num_samples=len(subset_indices))
            except Exception as e:
                print(f"Error creating dataset: {e}")
        
        except Exception as e:
            print(f"Error setting up attention analysis: {e}")
            print("You can modify this script to load your specific test data format.")