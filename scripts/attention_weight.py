import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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

if __name__ == "__main__":
    model_path = "output/package_encoder/package_encoder.pt"
    data_path = "data/processed/package_info.pkl"
    mappings_path = "data/processed/package_mappings.pkl"
    output_dir = "output/package_encoder/attention_visualization"
    
    visualize_attention_weights(model_path, data_path, mappings_path, output_dir)