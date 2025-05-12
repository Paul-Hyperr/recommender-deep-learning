#!/usr/bin/env python
"""
Analysis script for comparing FastText and LLM Package Encoders with Dutch travel packages

This script loads travel package data from a parquet file and compares 
the performance of FastText and LLM-based package encoders.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import time
import json
import argparse
from tqdm import tqdm
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import encoders
from models.package_encoder import PackageEncoder
from models.llm_package_encoder import LLMPackageEncoder

def preprocess_dutch_title(title):
    """
    Preprocess Dutch travel package titles
    
    Args:
        title (str): Original title
    
    Returns:
        str: Cleaned title
    """
    if not isinstance(title, str):
        return ""
        
    # Remove emojis and other non-text characters
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    title = emoji_pattern.sub(r'', title)
    
    # Standardize Dutch abbreviations
    abbreviation_map = {
        'incl.': 'inclusief',
        'o.b.v.': 'op basis van',
        'excl.': 'exclusief',
        'i.c.m.': 'in combinatie met',
        'bijv.': 'bijvoorbeeld',
        'evt.': 'eventueel',
        'max.': 'maximaal',
        'min.': 'minimaal',
    }
    
    # Replace abbreviations
    for abbr, full_form in abbreviation_map.items():
        title = title.replace(abbr, full_form)
    
    # Remove extra whitespace
    title = ' '.join(title.split())
    
    return title

def load_data_from_parquet(parquet_path, limit=100):
    """Load travel package data from parquet file"""
    print(f"Loading package data from {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        
        # Ensure title column exists
        if 'title' not in df.columns:
            raise ValueError("Parquet file must contain 'title' column")
        
        # Add main_id column if it doesn't exist
        if 'main_id' not in df.columns:
            print("No 'main_id' column found, using row indices as main_id")
            df['main_id'] = df.index.astype(str)
        else:
            # Ensure main_id is a string for consistent handling
            df['main_id'] = df['main_id'].astype(str)
        
        # Take first 'limit' rows
        if limit > 0:
            df = df.head(limit)
        
        # Fill NaN values and ensure types
        df['title'] = df['title'].fillna("").astype(str)
        
        # Add placeholder values for country, category, and theme if not present
        if 'country' not in df.columns:
            df['country'] = 'Nederland'  # Default country
        
        if 'category' not in df.columns:
            # Assign placeholder categories based on title keywords
            def assign_category(title):
                title_lower = title.lower()
                if any(word in title_lower for word in ['hotel', 'resort', 'verblijf']):
                    return 'Hotel'
                elif any(word in title_lower for word in ['strand', 'beach', 'zee']):
                    return 'Beach'
                elif any(word in title_lower for word in ['stad', 'city', 'stedentrip']):
                    return 'City Trip'
                else:
                    return 'General'
            
            df['category'] = df['title'].apply(assign_category)
        
        if 'theme' not in df.columns:
            # Assign placeholder themes based on title keywords
            def assign_theme(title):
                title_lower = title.lower()
                if any(word in title_lower for word in ['familie', 'family', 'kinderen', 'kids']):
                    return 'Family'
                elif any(word in title_lower for word in ['luxe', 'luxury', 'premium']):
                    return 'Luxury'
                elif any(word in title_lower for word in ['romantisch', 'romantic', 'koppel']):
                    return 'Romantic'
                else:
                    return 'General'
            
            df['theme'] = df['title'].apply(assign_theme)
        
        print(f"Loaded {len(df)} packages. Sample titles:")
        for i, title in enumerate(df['title'].head(5)):
            print(f"  {i+1}. {title} (main_id: {df['main_id'].iloc[i]})")
        
        return df
    
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        raise

def create_mappings_from_df(df):
    """Create mappings from DataFrame for testing"""
    
    # Create mapping dictionaries with index 0 reserved for padding/unknown
    country_to_idx = {country: idx+1 for idx, country in enumerate(df['country'].unique())}
    category_to_idx = {category: idx+1 for idx, category in enumerate(df['category'].unique())}
    theme_to_idx = {theme: idx+1 for idx, theme in enumerate(df['theme'].unique())}
    
    # Create reverse mappings
    idx_to_country = {idx: country for country, idx in country_to_idx.items()}
    idx_to_category = {idx: category for category, idx in category_to_idx.items()}
    idx_to_theme = {idx: theme for theme, idx in theme_to_idx.items()}
    
    # Create package mapping (assuming id is position in dataframe)
    package_to_idx = {idx+1: idx+1 for idx in range(len(df))}
    idx_to_package = {idx: idx for idx, _ in package_to_idx.items()}
    
    # Combine mappings
    mappings = {
        'country_to_idx': country_to_idx,
        'category_to_idx': category_to_idx,
        'theme_to_idx': theme_to_idx,
        'idx_to_country': idx_to_country,
        'idx_to_category': idx_to_category,
        'idx_to_theme': idx_to_theme,
        'package_to_idx': package_to_idx,
        'idx_to_package': idx_to_package
    }
    
    print(f"Created mappings: {len(mappings['country_to_idx'])} countries, "
          f"{len(mappings['category_to_idx'])} categories, "
          f"{len(mappings['theme_to_idx'])} themes")
    
    return mappings

def process_embeddings(encoder, df, model_name, output_dir, batch_size=200):
    """
    Process embeddings for the dataset
    
    Args:
        encoder: The encoder to use (FastText or LLM)
        df: Dataframe containing package data
        model_name: Name of the model ('FastText' or 'LLM')
        output_dir: Directory to save outputs
        batch_size: Number of items to process in each batch
    
    Returns:
        dict: Dictionary of embeddings by main_id and metrics
    """
    print(f"\nProcessing {model_name} embeddings for {len(df)} packages...")
    
    # Preprocess titles
    df['preprocessed_title'] = df['title'].apply(preprocess_dutch_title)
    
    # Track processing time and API calls
    start_time = time.time()
    
    # Create embedding dictionary
    embeddings_dict = {}
    
    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc=f"Processing {model_name} Embeddings"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(df))
        
        batch_df = df.iloc[batch_start:batch_end]
        
        batch_titles = batch_df['preprocessed_title'].tolist()
        batch_main_ids = batch_df['main_id'].tolist()
        
        try:
            # Process batch
            with torch.no_grad():
                batch_embeddings = encoder.process_batch_titles(batch_titles, batch_main_ids)
            
            # Store embeddings by main_id
            for i, main_id in enumerate(batch_main_ids):
                embedding = batch_embeddings[i].detach().numpy()
                embeddings_dict[main_id] = embedding
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            # Continue with next batch
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Embedding metrics
    if embeddings_dict:
        # Convert to numpy array for analysis
        embeddings_array = np.array(list(embeddings_dict.values()))
        
        # Compute basic statistics
        norms = np.linalg.norm(embeddings_array, axis=1)
        
        # Compute pairwise similarities for a sample (too slow for all pairs)
        sample_size = min(100, len(embeddings_dict))
        sample_indices = np.random.choice(len(embeddings_dict), sample_size, replace=False)
        sample_embeddings = embeddings_array[sample_indices]
        
        # Normalize for cosine similarity
        sample_norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        normalized_sample = sample_embeddings / sample_norms
        
        # Compute pairwise similarities
        similarities = cosine_similarity(normalized_sample)
        np.fill_diagonal(similarities, 0)  # Zero out self-similarities
        
        metrics = {
            'processing_time': processing_time,
            'processing_time_per_item': processing_time / len(df),
            'embedding_dim': embeddings_array.shape[1],
            'mean_norm': float(norms.mean()),
            'std_norm': float(norms.std()),
            'mean_similarity': float(similarities.mean()),
            'std_similarity': float(similarities.std()),
            'min_similarity': float(similarities.min()),
            'max_similarity': float(similarities.max()),
        }
    else:
        metrics = {
            'processing_time': processing_time,
            'processing_time_per_item': processing_time / len(df) if len(df) > 0 else 0,
            'error': 'No embeddings generated'
        }
    
    # Save embeddings for later use
    embeddings_path = os.path.join(output_dir, f"{model_name.lower()}_embeddings.npz")
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings_dict,
        main_ids=list(embeddings_dict.keys())
    )
    
    print(f"{model_name} embeddings saved to {embeddings_path}")
    print(f"Total processing time: {processing_time:.2f}s ({processing_time/len(df):.4f}s per item)")
    
    # Visualize embeddings
    if len(embeddings_dict) > 10:
        visualize_embeddings(embeddings_array, output_dir, model_name)
    
    return {
        'embeddings': embeddings_dict,
        'metrics': metrics
    }

def visualize_embeddings(embeddings, output_dir, model_name):
    """
    Create t-SNE visualization of embeddings
    
    Args:
        embeddings: Numpy array of embeddings
        output_dir: Output directory
        model_name: Name of the model
    """
    # Sample if too many embeddings (t-SNE is slow for large datasets)
    max_samples = 1000
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings_sample = embeddings[indices]
    else:
        embeddings_sample = embeddings
    
    try:
        # Standardize embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings_sample)
        
        # Compute t-SNE (adjust perplexity based on sample size)
        perplexity = min(30, len(scaled_embeddings) - 1)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=42
        )
        
        # Apply t-SNE transformation
        tsne_results = tsne.fit_transform(scaled_embeddings)
        
        # Plot results
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.6, s=20)
        plt.title(f't-SNE visualization of {model_name} embeddings')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'{model_name.lower()}_embeddings_tsne.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"t-SNE visualization saved to {output_path}")
    
    except Exception as e:
        print(f"Error creating t-SNE visualization: {e}")

def compare_encoders(df, fasttext_encoder, llm_encoder, output_dir):
    """
    Compare both encoders on the dataset
    
    Args:
        df: DataFrame with package data
        fasttext_encoder: FastText encoder
        llm_encoder: LLM encoder
        output_dir: Output directory
    
    Returns:
        dict: Comparison results
    """
    # Process embeddings with both encoders
    fasttext_results = process_embeddings(
        fasttext_encoder, df, "FastText", output_dir
    )
    
    llm_results = process_embeddings(
        llm_encoder, df, "LLM", output_dir
    )
    
    # Compare metrics
    if 'metrics' in fasttext_results and 'metrics' in llm_results:
        fasttext_metrics = fasttext_results['metrics']
        llm_metrics = llm_results['metrics']
        
        comparison = {
            'processing_time_ratio': llm_metrics['processing_time'] / fasttext_metrics['processing_time'] 
                if fasttext_metrics['processing_time'] > 0 else float('inf'),
            'fasttext_metrics': fasttext_metrics,
            'llm_metrics': llm_metrics
        }
        
        # Create comparison visualization
        create_comparison_visualization(fasttext_metrics, llm_metrics, output_dir)
        
        # Generate comparison report
        generate_comparison_report(comparison, output_dir)
    else:
        comparison = {
            'error': 'Missing metrics for one or both encoders'
        }
    
    return comparison

def create_comparison_visualization(fasttext_metrics, llm_metrics, output_dir):
    """
    Create comparative visualizations
    
    Args:
        fasttext_metrics: Metrics for FastText
        llm_metrics: Metrics for LLM
        output_dir: Output directory
    """
    # Processing time comparison
    plt.figure(figsize=(10, 6))
    
    metrics = [
        ('processing_time_per_item', 'Processing Time (s/item)'),
        ('mean_similarity', 'Mean Similarity'),
        ('std_similarity', 'Similarity Std Dev')
    ]
    
    # Filter metrics that exist in both
    valid_metrics = [m for m, _ in metrics if m in fasttext_metrics and m in llm_metrics]
    metric_labels = [label for m, label in metrics if m in valid_metrics]
    
    if not valid_metrics:
        print("No common metrics to visualize")
        return
    
    # Prepare data
    x = np.arange(len(valid_metrics))
    fasttext_values = [fasttext_metrics[m] for m in valid_metrics]
    llm_values = [llm_metrics[m] for m in valid_metrics]
    
    # Plot bars
    width = 0.35
    plt.bar(x - width/2, fasttext_values, width, label='FastText')
    plt.bar(x + width/2, llm_values, width, label='LLM')
    
    # Add labels and legend
    plt.ylabel('Value')
    plt.title('Encoder Comparison')
    plt.xticks(x, metric_labels)
    plt.legend()
    
    # Save figure
    output_path = os.path.join(output_dir, 'encoder_comparison.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Comparison visualization saved to {output_path}")

def generate_comparison_report(comparison, output_dir):
    """
    Generate a comprehensive comparison report
    
    Args:
        comparison: Comparison results
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# FastText vs LLM Encoder Comparison\n\n")
        
        # Performance comparison
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | FastText | LLM | Ratio (LLM/FastText) |\n")
        f.write("|--------|----------|-----|----------------------|\n")
        
        ft_metrics = comparison['fasttext_metrics']
        llm_metrics = comparison['llm_metrics']
        
        # Processing time
        ft_time = ft_metrics.get('processing_time_per_item', 0)
        llm_time = llm_metrics.get('processing_time_per_item', 0)
        ratio = llm_time / ft_time if ft_time > 0 else float('inf')
        
        f.write(f"| Processing time per item | {ft_time:.4f}s | {llm_time:.4f}s | {ratio:.2f}x |\n")
        
        # Other metrics that both have
        common_metrics = set(ft_metrics.keys()).intersection(set(llm_metrics.keys()))
        common_metrics = [m for m in common_metrics if m != 'processing_time' and m != 'processing_time_per_item']
        
        for metric in common_metrics:
            ft_value = ft_metrics[metric]
            llm_value = llm_metrics[metric]
            
            # Only calculate ratio for non-zero values
            if isinstance(ft_value, (int, float)) and isinstance(llm_value, (int, float)) and ft_value != 0:
                ratio = llm_value / ft_value
                f.write(f"| {metric} | {ft_value:.4f} | {llm_value:.4f} | {ratio:.2f}x |\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        
        # Processing time comparison
        if ratio > 10:
            f.write(f"LLM encoding is significantly slower ({ratio:.2f}x) than FastText. ")
        elif ratio > 2:
            f.write(f"LLM encoding is moderately slower ({ratio:.2f}x) than FastText. ")
        else:
            f.write(f"LLM encoding is comparable in speed ({ratio:.2f}x) to FastText. ")
        
        # Similarity analysis if available
        if 'mean_similarity' in common_metrics:
            ft_sim = ft_metrics['mean_similarity']
            llm_sim = llm_metrics['mean_similarity']
            
            if llm_sim > ft_sim * 1.1:
                f.write("LLM embeddings show higher average similarity between packages, which may indicate better clustering of related items.")
            elif ft_sim > llm_sim * 1.1:
                f.write("FastText embeddings show higher average similarity between packages, which may indicate better clustering of related items.")
            else:
                f.write("Both encoders show similar average similarity between package embeddings.")
        
        # Recommendations
        f.write("\n\n### Recommendations\n\n")
        
        if ratio > 20:
            f.write("Given the significant speed difference, FastText is recommended for real-time applications where latency is critical. "
                   "LLM should be reserved for offline processing or when embedding quality requirements outweigh performance considerations.")
        elif ratio > 5:
            f.write("Consider using a hybrid approach: FastText for real-time processing and LLM for offline tasks or when higher quality embeddings are required.")
        else:
            f.write("Both encoders offer reasonable performance trade-offs. Choose based on specific application needs and quality requirements.")
        
        # API cost considerations for LLM
        f.write("\n\n### API Cost Considerations\n\n")
        f.write("When using the LLM encoder in production, remember to factor in OpenAI API costs. "
               "The implementation includes caching to minimize API calls, but initial embedding generation "
               "will incur costs based on the number of tokens processed.")
        
        # Suggest next steps
        f.write("\n\n## Next Steps\n\n")
        f.write("1. **Evaluate embedding quality**: Test both encoders on downstream tasks like recommendations or search to evaluate real-world performance.\n")
        f.write("2. **Fine-tune parameters**: Adjust embedding dimensions, attention mechanisms, or other parameters to optimize performance.\n")
        f.write("3. **Implement hybrid approach**: Consider using FastText for real-time processing and LLM for offline batch processing.\n")
        
    print(f"Comparison report saved to {report_path}")
    return report_path

def initialize_encoders(mappings, fasttext_model_path, api_key, embedding_dim=256, hidden_dim=256):
    """
    Initialize both encoders
    
    Args:
        mappings (dict): Mapping dictionaries
        fasttext_model_path (str): Path to FastText model
        api_key (str): OpenAI API key
        embedding_dim (int): Dimension of embeddings
        hidden_dim (int): Dimension of hidden layers
    
    Returns:
        tuple: FastText and LLM encoders
    """
    # Extract vocab sizes
    package_vocab_size = len(mappings['package_to_idx']) + 1  # +1 for padding
    country_vocab_size = len(mappings['country_to_idx']) + 1
    category_vocab_size = len(mappings['category_to_idx']) + 1
    theme_vocab_size = len(mappings['theme_to_idx']) + 1
    
    print("Initializing FastText-based PackageEncoder...")
    fasttext_encoder = PackageEncoder(
        package_vocab_size=package_vocab_size,
        country_vocab_size=country_vocab_size,
        category_vocab_size=category_vocab_size,
        theme_vocab_size=theme_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        fasttext_model_path=fasttext_model_path
    )
    
    print("Initializing LLM-based PackageEncoder...")
    llm_encoder = LLMPackageEncoder(
        package_vocab_size=package_vocab_size,
        country_vocab_size=country_vocab_size,
        category_vocab_size=category_vocab_size,
        theme_vocab_size=theme_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        api_key=api_key,
        embedding_model="text-embedding-3-large"  # More expensive version
    )
    
    return fasttext_encoder, llm_encoder

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare FastText and LLM Package Encoders")
    
    parser.add_argument("--parquet_path", type=str, default="data/feed.parquet",
                      help="Path to parquet file with package data")
    parser.add_argument("--fasttext_model", type=str, default="data/cc.nl.300.bin",
                      help="Path to FastText model for Dutch")
    parser.add_argument("--api_key", type=str, default=None,
                      help="OpenAI API key (if not provided, will use OPENAI_API_KEY env var)")
    parser.add_argument("--output_dir", type=str, default="output/llm_encoder_comparison",
                      help="Directory to save results")
    parser.add_argument("--limit", type=int, default=100,
                      help="Number of packages to process (0 for all)")
    parser.add_argument("--batch_size", type=int, default=50,
                      help="Batch size for processing")
    parser.add_argument("--embedding_dim", type=int, default=256,
                      help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                      help="Hidden layer dimension")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key or OPENAI_API_KEY environment variable")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data from parquet file
    df = load_data_from_parquet(args.parquet_path, limit=args.limit)
    
    # Create mappings from data
    mappings = create_mappings_from_df(df)
    
    # Initialize encoders
    fasttext_encoder, llm_encoder = initialize_encoders(
        mappings,
        args.fasttext_model,
        api_key,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim
    )
    
    # Compare encoders
    comparison = compare_encoders(
        df,
        fasttext_encoder,
        llm_encoder,
        args.output_dir
    )
    
    # Save full comparison results
    results_path = os.path.join(args.output_dir, 'comparison_results.json')
    
    # Convert to JSON-compatible format (remove numpy arrays and tensors)
    json_safe_comparison = {}
    for key, value in comparison.items():
        if key != 'fasttext_embeddings' and key != 'llm_embeddings':
            if isinstance(value, dict):
                # Convert nested dictionaries
                json_safe_comparison[key] = {
                    k: float(v) if isinstance(v, (np.number, np.ndarray)) else v
                    for k, v in value.items()
                }
            else:
                json_safe_comparison[key] = float(value) if isinstance(value, (np.number, np.ndarray)) else value
    
    with open(results_path, 'w') as f:
        json.dump(json_safe_comparison, f, indent=2)
    
    print(f"Full comparison results saved to {results_path}")
    
    # Print summary comparison
    if 'processing_time_ratio' in comparison:
        print(f"\nSummary: LLM encoder is {comparison['processing_time_ratio']:.2f}x slower than FastText")
    
    print("\nAnalysis complete! Check the output directory for detailed results and visualizations.")

if __name__ == "__main__":
    main()