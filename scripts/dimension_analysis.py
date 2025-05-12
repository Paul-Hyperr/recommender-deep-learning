import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.package_encoder import PackageEncoder

def analyze_dimensionality(representations_path, package_ids_path, mappings_path, output_dir):
    """
    Analyze the dimensionality of package representations
    
    Args:
        representations_path: Path to saved package representations
        package_ids_path: Path to saved package IDs
        mappings_path: Path to saved mappings
        output_dir: Output directory for analysis results
    """
    print(f"Loading package representations from {representations_path}")
    representations = np.load(representations_path)
    
    with open(package_ids_path, 'rb') as f:
        package_ids = pickle.load(f)
    
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    # Extract mappings
    category_to_idx = mappings['category_to_idx']
    idx_to_category = {idx: cat for cat, idx in category_to_idx.items()}
    
    print(f"Loaded {len(representations)} package representations with dimension {representations.shape[1]}")
    
    # 1. PCA Analysis to determine optimal dimensionality
    print("Performing PCA analysis...")
    pca = PCA()
    pca.fit(representations)
    
    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find dimensions needed for different variance thresholds
    variance_thresholds = [0.75, 0.85, 0.90, 0.95, 0.99]
    dims_needed = {}
    for threshold in variance_thresholds:
        dims = np.argmax(cumulative_variance >= threshold) + 1
        dims_needed[threshold] = dims
        print(f"Dimensions needed for {threshold*100:.0f}% variance: {dims}")
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, marker='o', markersize=3)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA: Cumulative Explained Variance vs. Number of Components')
    plt.grid(True)
    
    # Add horizontal lines for variance thresholds
    for threshold in variance_thresholds:
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.3)
        plt.text(representations.shape[1]*0.8, threshold+0.01, f'{threshold*100:.0f}%')
    
    # Add vertical lines for dimensions needed
    for threshold, dims in dims_needed.items():
        plt.axvline(x=dims, color='g', linestyle='--', alpha=0.3)
        plt.text(dims+5, 0.5, f'{dims} dims')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    
    # 2. Train models with different embedding dimensions
    embedding_dims = [64, 128, 256, 512]
    training_times = []
    inference_times = []
    
    print("\nTesting different embedding dimensions...")
    
    # Use a subset of data for faster testing
    test_size = min(1000, len(representations))
    test_titles = ["Test title"] * test_size
    test_countries = torch.ones(test_size, dtype=torch.long)
    test_categories = torch.ones(test_size, dtype=torch.long)
    test_themes = torch.ones(test_size, dtype=torch.long)
    
    for dim in embedding_dims:
        print(f"\nTesting embedding dimension: {dim}")
        
        # Create encoder with this dimension
        start_time = time.time()
        encoder = PackageEncoder(
            package_vocab_size=100,
            country_vocab_size=100,
            category_vocab_size=100,
            theme_vocab_size=100,
            embedding_dim=dim,
            hidden_dim=dim*2,
            fasttext_model_path="data/cc.nl.300.bin"
        )
        init_time = time.time() - start_time
        print(f"Initialization time: {init_time:.4f} seconds")
        
        # Count parameters
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            _ = encoder(
                test_titles[:10], 
                test_countries[:10], 
                test_categories[:10], 
                test_themes[:10]
            )
        inference_time = (time.time() - start_time) / 10  # Per item
        print(f"Inference time per item: {inference_time*1000:.2f} ms")
        
        training_times.append(init_time)
        inference_times.append(inference_time)
    
    # Plot timing results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(embedding_dims, training_times, marker='o')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Initialization Time (s)')
    plt.title('Model Initialization Time vs. Embedding Dimension')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(embedding_dims, [t*1000 for t in inference_times], marker='o')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Inference Time per Item (ms)')
    plt.title('Inference Time vs. Embedding Dimension')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_timing.png'))
    
    # 3. Visualize representations in 2D using t-SNE
    print("\nVisualizing representations with t-SNE...")
    
    # Use PCA first to reduce dimensions to 50 for faster t-SNE
    pca = PCA(n_components=min(50, representations.shape[1]))
    reduced_representations = pca.fit_transform(representations)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(reduced_representations[:2000])  # Using a subset for speed
    
    # Get categories for the subset
    cats = []
    for package_id in package_ids[:2000]:
        # This is a placeholder - you'd need to get the actual category for each package
        # For now, let's use a random category
        cats.append(np.random.randint(1, len(category_to_idx) + 1))
    
    # Plot t-SNE results colored by category
    plt.figure(figsize=(12, 10))
    unique_cats = set(cats)
    
    for cat in unique_cats:
        idx = [i for i, c in enumerate(cats) if c == cat]
        plt.scatter(
            tsne_results[idx, 0], 
            tsne_results[idx, 1], 
            label=idx_to_category.get(cat, f"Category {cat}"),
            alpha=0.7
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Visualization of Package Representations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    
    print(f"\nDimensionality analysis completed. Results saved to {output_dir}")
    
    return dims_needed, training_times, inference_times

if __name__ == "__main__":
    representations_path = "output/package_encoder/package_representations.npy"
    package_ids_path = "output/package_encoder/package_ids.pkl"
    mappings_path = "data/processed/package_mappings.pkl"
    output_dir = "output/package_encoder/dimension_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    dims_needed, training_times, inference_times = analyze_dimensionality(
        representations_path, package_ids_path, mappings_path, output_dir
    )