import pandas as pd
import numpy as np
import fasttext
import spacy
from tqdm import tqdm
import pickle
import os

def prepare_package_data(data_path, fasttext_model_path="cc.nl.300.bin", output_dir="data/processed"):
    """
    Prepare package data for the NATR model by extracting and preprocessing package information
    
    Args:
        data_path: Path to the data file (parquet)
        fasttext_model_path: Path to the FastText model
        output_dir: Output directory for processed data
    """
    print(f"Preparing package data from {data_path}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load FastText model
    print(f"Loading FastText model from {fasttext_model_path}...")
    fasttext_model = fasttext.load_model(fasttext_model_path)
    
    # Load SpaCy's Dutch language model for stopwords
    nlp = spacy.blank("nl")
    # Get Dutch stopwords
    dutch_stopwords = nlp.Defaults.stop_words
    custom_stopwords = {"incl.", "o.b.v.", "bijvoorbeeld", "minstens", "het", "een", "Incl.", 
                        "&", "direct", "regio", "o.a.", "én", "easy", "going", "'s", "van", "extra"}
    stopwords = dutch_stopwords.union(custom_stopwords)
    
    # Function to remove stopwords from a sentence
    def remove_stopwords(sentence, stopword_list):
        if not isinstance(sentence, str):
            return ""
        tokens = sentence.lower().split()  # Split into words
        filtered_tokens = [word for word in tokens if word not in stopword_list]  # Remove stopwords
        return " ".join(filtered_tokens)
    
    # Load data
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    
    # Extract unique packages
    print("Extracting unique packages...")
    package_info = df[['main_id', 'title', 'country', 'category', 'theme', 'min_price']].drop_duplicates('main_id')
    
    # Remove stopwords from titles
    print("Processing titles...")
    package_info['filtered_title'] = package_info['title'].apply(lambda x: remove_stopwords(x, stopwords))
    
    # Generate FastText embeddings for filtered titles
    print("Generating FastText embeddings for titles...")
    embeddings = {}
    for idx, row in tqdm(package_info.iterrows(), total=len(package_info), desc="Embedding titles"):
        if row['filtered_title']:
            embeddings[row['main_id']] = fasttext_model.get_sentence_vector(row['filtered_title'])
        else:
            # Use zero vector for empty titles
            embeddings[row['main_id']] = np.zeros(fasttext_model.get_dimension())
    
    # Add embeddings to package info
    package_info['title_embedding'] = package_info['main_id'].map(embeddings)
    
    # Save processed data
    output_path = os.path.join(output_dir, "package_info.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(package_info, f)
    
    print(f"Saved processed package data to {output_path}")
    print(f"Total packages: {len(package_info)}")
    
    # Create mappings
    print("Creating mappings...")
    
    # Package ID to index mapping
    package_to_idx = {package_id: i+1 for i, package_id in enumerate(package_info['main_id'].unique())}
    
    # Country to index mapping
    countries = package_info['country'].dropna().unique()
    country_to_idx = {country: i+1 for i, country in enumerate(countries)}
    
    # Category to index mapping
    categories = package_info['category'].dropna().unique()
    category_to_idx = {category: i+1 for i, category in enumerate(categories)}
    
    # Theme to index mapping
    themes = package_info['theme'].dropna().unique()
    theme_to_idx = {theme: i+1 for i, theme in enumerate(themes)}
    
    # Save mappings
    mappings = {
        'package_to_idx': package_to_idx,
        'country_to_idx': country_to_idx,
        'category_to_idx': category_to_idx,
        'theme_to_idx': theme_to_idx
    }
    
    mappings_path = os.path.join(output_dir, "package_mappings.pkl")
    with open(mappings_path, 'wb') as f:
        pickle.dump(mappings, f)
    
    print(f"Saved mappings to {mappings_path}")
    print(f"Package vocabulary size: {len(package_to_idx)}")
    print(f"Country vocabulary size: {len(country_to_idx)}")
    print(f"Category vocabulary size: {len(category_to_idx)}")
    print(f"Theme vocabulary size: {len(theme_to_idx)}")
    
    return package_info, mappings

if __name__ == "__main__":
    data_path = "data/bookit_events_data_13_months.parquet"
    package_info, mappings = prepare_package_data(data_path)