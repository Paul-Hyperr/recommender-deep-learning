"""
NATR inference script for making recommendations
"""

import torch
import json
import numpy as np
import pandas as pd
import sys
import os
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor


class NATRRecommender:
    """NATR model for inference with availability and purchase history filtering"""
    
    def __init__(self, model_info_path='model_info.json', availability_data_path: Optional[str] = None, exclude_prev_purchases: bool = False, exclude_recent_months: Optional[int] = None):
        """Initialize recommender from saved model
        
        Args:
            model_info_path: Path to model info JSON file
            availability_data_path: Optional path to availability parquet file
            exclude_prev_purchases: Whether to exclude previously bought packages from recommendations
            exclude_recent_months: If set, only exclude purchases from the last N months instead of all history
        """
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Store configuration
        self.exclude_prev_purchases = exclude_prev_purchases
        self.exclude_recent_months = exclude_recent_months
        
        # Initialize tracking for purchase exclusions
        self.total_purchase_exclusions = 0
        self.total_recommendations_requested = 0
        # Track exclusions specifically for users with test purchases
        self.test_user_purchase_exclusions = 0
        self.test_users_processed = 0
        # Track ranking of excluded purchases
        self.excluded_purchase_ranks = []  # List of ranks where excluded purchases appeared
        
        # Get split date if available
        self.split_date = self.model_info.get('split_date')
        if self.split_date:
            print(f"Model trained with split date: {self.split_date}")
        else:
            print("Warning: No split date found in model info - purchase history will include all data")
        
        # Initialize availability data
        self.availability_data = None
        self.package_availability_days = {}
        if availability_data_path:
            self.load_availability_data(availability_data_path)
        
        # Initialize processors
        # Determine embedding model based on title_embedding_dim
        title_embedding_dim = self.model_info['config'].get('title_embedding_dim', 1536)
        if title_embedding_dim == 1536:
            embedding_model = 'text-embedding-3-small'
        elif title_embedding_dim == 3072:
            embedding_model = 'text-embedding-3-large'
        else:
            # Default to small if unclear
            embedding_model = 'text-embedding-3-small'
            print(f"Warning: Unexpected title_embedding_dim {title_embedding_dim}, using {embedding_model}")
        
        self.package_processor = PackageProcessor(
            feed_data_path=self.model_info.get('package_data_path', 'data/feed.parquet'),
            cache_dir='data/cache',
            load_coordinates=True,
            load_embeddings=True,
            embedding_model=embedding_model,
            use_reduced_embeddings=False  # Use full dimensions to match model
        )
        
        self.session_processor = SessionProcessor(
            event_data_path=self.model_info.get('event_data_path', 'data/bookit_events_13_months.parquet'),
            cache_dir='data/cache'
        )
        
        # Load data
        print("Loading data...")
        self.package_processor.load_data()
        self.session_processor.load_data()
        
        # Create mappings
        self.package_processor.create_mappings()
        self.session_processor.create_mappings()
        
        # Extract sessions for purchase history tracking
        self.session_processor.extract_sessions()
        
        # Get mappings
        self.user_to_idx = self.session_processor.get_idx_mappings()['user_to_idx']
        self.package_to_idx = self.session_processor.get_idx_mappings()['package_to_idx']
        self.idx_to_package = {v: k for k, v in self.package_to_idx.items()}
        
        # Create model
        print("Loading model...")
        config = NATRConfig(**self.model_info['config'])
        self.model = NATR(config)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_info['checkpoint_path'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Valid packages - if not in model_info, use all packages from data
        if 'valid_packages' in self.model_info and self.model_info['valid_packages']:
            self.valid_packages = set(self.model_info['valid_packages'])
        else:
            # Use all packages from the loaded data
            self.valid_packages = set(str(pkg_id) for pkg_id in self.package_to_idx.keys())
            print(f"Using all {len(self.valid_packages)} packages as valid packages")
        
        # Cache for purchase history to improve performance
        self._purchase_history_cache = {}
        self._build_purchase_history_cache()
        
        print("Recommender ready!")
    
    def _build_purchase_history_cache(self):
        """Build a cache of purchase history for all users for better performance"""
        print("Building purchase history cache...")
        
        # Convert split date to datetime if available
        split_datetime = None
        recent_cutoff_datetime = None
        
        if self.split_date:
            try:
                split_datetime = pd.to_datetime(self.split_date)
                print(f"Filtering purchase history to only include purchases before {self.split_date}")
                
                # If exclude_recent_months is set, calculate cutoff date
                if self.exclude_recent_months:
                    recent_cutoff_datetime = split_datetime - pd.DateOffset(months=self.exclude_recent_months)
                    print(f"Only excluding purchases from last {self.exclude_recent_months} months (after {recent_cutoff_datetime.strftime('%Y-%m-%d')})")
                
            except:
                print(f"Warning: Could not parse split date {self.split_date}, including all purchases")
        
        for session in self.session_processor.sessions:
            user_id = session['user_id']
            
            # Check if session is before split date
            if split_datetime:
                session_end_time = pd.to_datetime(session.get('end_time'))
                if session_end_time >= split_datetime:
                    # Skip sessions that are in the test period
                    continue
                
                # If we're only excluding recent months, skip older sessions
                if recent_cutoff_datetime and session_end_time < recent_cutoff_datetime:
                    # This session is too old to be excluded
                    continue
            
            # Initialize user's purchase set if not exists
            if user_id not in self._purchase_history_cache:
                self._purchase_history_cache[user_id] = set()
            
            # Add all purchased packages from this session
            for i, (pkg_id, is_purchase) in enumerate(zip(session['package_ids'], session['is_purchase'])):
                if is_purchase == 1:  # This was a purchase
                    self._purchase_history_cache[user_id].add(str(pkg_id))
        
        print(f"Built purchase history cache for {len(self._purchase_history_cache)} users")
        
        # Report some statistics
        if self._purchase_history_cache:
            purchase_counts = [len(purchases) for purchases in self._purchase_history_cache.values()]
            avg_purchases = sum(purchase_counts) / len(purchase_counts)
            max_purchases = max(purchase_counts)
            print(f"Average purchases per user: {avg_purchases:.1f}, Max: {max_purchases}")
    
    def load_availability_data(self, availability_data_path: str):
        """Load availability data from parquet file and count available days per package
        
        Args:
            availability_data_path: Path to availability parquet file with columns: main_id, date
        """
        try:
            print(f"Loading availability data from {availability_data_path}...")
            self.availability_data = pd.read_parquet(availability_data_path)
            
            # Ensure main_id is string
            self.availability_data['main_id'] = self.availability_data['main_id'].astype(str)
            
            # Count distinct dates per package
            availability_counts = self.availability_data.groupby('main_id')['date'].nunique()
            self.package_availability_days = availability_counts.to_dict()
            
            print(f"Loaded availability data for {len(self.package_availability_days)} packages")
            print(f"Average available days per package: {np.mean(list(self.package_availability_days.values())):.1f}")
            print(f"Max available days: {max(self.package_availability_days.values())}")
            
        except Exception as e:
            print(f"Error loading availability data: {e}")
            self.availability_data = None
            self.package_availability_days = {}
    
    def prepare_user_data(self, user_id: str) -> Optional[Dict]:
        """Prepare user data for recommendation"""
        
        # Get user sessions
        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            print(f"Unknown user: {user_id}")
            return None
        
        # Get user's sessions from session processor
        user_sessions = []
        for session in self.session_processor.sessions:
            if session['user_id'] == user_id:
                user_sessions.append(session)
        
        if not user_sessions:
            print(f"No sessions found for user: {user_id}")
            return None
        
        # Sort by time
        user_sessions.sort(key=lambda x: x['end_time'])
        
        # Extract short-term and long-term
        if len(user_sessions) == 1:
            short_term = user_sessions[0]
            long_term = []
        else:
            short_term = user_sessions[-1]
            long_term = user_sessions[:-1]
        
        # Prepare features
        short_term_packages = []
        short_term_events = []
        
        for i, pkg_id in enumerate(short_term['package_ids']):
            if str(pkg_id) in self.valid_packages:
                short_term_packages.append(pkg_id)
                short_term_events.append(short_term['event_types'][i])
        
        long_term_packages = []
        long_term_events = []
        
        for session in long_term:
            for i, pkg_id in enumerate(session['package_ids']):
                if str(pkg_id) in self.valid_packages:
                    long_term_packages.append(pkg_id)
                    long_term_events.append(session['event_types'][i])
        
        return {
            'user_id': user_id,
            'user_idx': user_idx,
            'short_term_packages': short_term_packages[-10:],  # Max 10
            'short_term_events': short_term_events[-10:],
            'long_term_packages': long_term_packages[-20:],  # Max 20
            'long_term_events': long_term_events[-20:]
        }
    
    def get_user_purchase_history(self, user_id: str) -> set:
        """Get all packages that a user has previously purchased (cached for performance)"""
        return self._purchase_history_cache.get(user_id, set())
    
    def filter_available_packages(self, package_ids: List[str], available_packages: Optional[set] = None) -> List[str]:
        """
        Filter packages to only include those that are available
        
        Args:
            package_ids: List of package IDs to filter
            available_packages: Set of available package IDs. If None, uses self.valid_packages
        
        Returns:
            List of available package IDs
        """
        if available_packages is None:
            available_packages = self.valid_packages
        
        return [pkg_id for pkg_id in package_ids if str(pkg_id) in available_packages]
    
    def filter_unpurchased_packages(self, package_ids: List[str], user_id: str) -> List[str]:
        """
        Filter packages to exclude those already purchased by the user
        
        Args:
            package_ids: List of package IDs to filter
            user_id: User ID to check purchase history for
        
        Returns:
            List of package IDs not previously purchased by the user
        """
        purchased_packages = self.get_user_purchase_history(user_id)
        return [pkg_id for pkg_id in package_ids if str(pkg_id) not in purchased_packages]
    
    def _move_to_device(self, batch, device):
        """Move batch data to specified device"""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {key: self._move_to_device(value, device) for key, value in batch.items()}
        else:
            return batch
    
    def recommend(self, user_id: str, top_k: int = 10, 
                  available_packages: Optional[set] = None,
                  min_available_days: int = 5,
                  exclude_purchased: Optional[bool] = None,
                  max_candidates: int = None,
                  is_test_user: bool = False) -> List[Dict]:
        """
        Get recommendations for a user with availability and purchase history filtering
        Ensures exactly top_k recommendations by getting enough candidates after filtering
        
        Args:
            user_id: User ID to get recommendations for
            top_k: Number of recommendations to return
            available_packages: Set of currently available package IDs. If None, uses all valid packages
            min_available_days: Minimum number of available days required (default: 5)
            exclude_purchased: Whether to exclude packages the user has already purchased
            max_candidates: Maximum number of candidates to consider (default: all valid packages)
            is_test_user: Whether this user has purchases in the test data (for tracking purposes)
        
        Returns:
            List of exactly top_k recommendation dictionaries (or fewer if not enough valid packages)
        """
        
        # Prepare user data
        user_data = self.prepare_user_data(user_id)
        if user_data is None:
            return []
        
        # Create batch
        batch = self._create_batch(user_data)
        
        # Determine whether to exclude purchased packages
        if exclude_purchased is None:
            exclude_purchased = self.exclude_prev_purchases
        
        # Get user's purchase history upfront for filtering
        purchased_packages = set()
        if exclude_purchased:
            purchased_packages = self.get_user_purchase_history(user_id)
        
        # Determine how many candidates we need to get
        if max_candidates is None:
            max_candidates = len(self.valid_packages)
        
        # Get all predictions (we'll filter and select top_k from these)
        with torch.no_grad():
            # Move batch to same device as model
            device = next(self.model.parameters()).device
            batch = self._move_to_device(batch, device)
            
            outputs = self.model(batch)
            predictions = outputs['predictions']
        
        # Get all scores and indices, sorted by score
        all_scores, all_indices = torch.sort(predictions[0], descending=True)
        
        # Filter candidates one by one until we have top_k valid recommendations
        recommendations = []
        processed_count = 0
        purchase_exclusions = 0
        
        for i in range(min(max_candidates, len(all_indices))):
            if len(recommendations) >= top_k:
                break
                
            pkg_idx = all_indices[i].item()
            pkg_id = self.idx_to_package.get(pkg_idx)
            processed_count += 1
            
            # Skip if package is not in valid packages
            if not pkg_id or str(pkg_id) not in self.valid_packages:
                continue
            
            pkg_id_str = str(pkg_id)
            
            # Apply availability filtering
            if available_packages is not None and pkg_id_str not in available_packages:
                continue
            
            # Apply min available days filtering
            if min_available_days > 0 and self.package_availability_days:
                available_days = self.package_availability_days.get(pkg_id_str, 0)
                if available_days < min_available_days:
                    continue
            
            # Apply purchase history filtering for this specific user
            if exclude_purchased and pkg_id_str in purchased_packages:
                purchase_exclusions += 1
                # Track the rank where this excluded purchase would have appeared
                if is_test_user:
                    current_rank = i + 1  # 1-indexed rank in the full sorted list
                    self.excluded_purchase_ranks.append(current_rank)
                continue
            
            # Get package info
            pkg_info = self.package_processor.get_package_features(pkg_id_str)
            if not pkg_info:
                continue
            
            # Add to recommendations
            rec_dict = {
                'rank': len(recommendations) + 1,
                'package_id': pkg_id_str,
                'score': all_scores[i].item(),
                'title': pkg_info.get('title', ''),
                'city': pkg_info.get('city', ''),
                'country': pkg_info.get('country', ''),
                'theme': pkg_info.get('theme', ''),
                'category': pkg_info.get('category', ''),
                'price': pkg_info.get('price', 0)
            }
            
            # Add availability days if we have that data
            if self.package_availability_days:
                rec_dict['available_days'] = self.package_availability_days.get(pkg_id_str, 0)
            
            recommendations.append(rec_dict)
        
        # Update aggregate tracking
        self.total_purchase_exclusions += purchase_exclusions
        self.total_recommendations_requested += 1
        
        # Track exclusions for test users specifically
        if is_test_user:
            self.test_user_purchase_exclusions += purchase_exclusions
            self.test_users_processed += 1
        
        # Print aggregate statistics every 500 users
        if exclude_purchased and self.total_recommendations_requested % 500 == 0:
            avg_exclusions = self.total_purchase_exclusions / self.total_recommendations_requested
            print(f"Purchase exclusion stats:")
            print(f"  Total: {self.total_purchase_exclusions} exclusions across {self.total_recommendations_requested} users (avg: {avg_exclusions:.1f}/user)")
            
            if self.test_users_processed > 0:
                test_avg_exclusions = self.test_user_purchase_exclusions / self.test_users_processed
                print(f"  Test users: {self.test_user_purchase_exclusions} exclusions across {self.test_users_processed} test users (avg: {test_avg_exclusions:.1f}/user)")
                
                # Analyze excluded purchase ranks
                if self.excluded_purchase_ranks:
                    import numpy as np
                    ranks = np.array(self.excluded_purchase_ranks)
                    print(f"  Excluded purchase ranks - Min: {ranks.min()}, Max: {ranks.max()}, Mean: {ranks.mean():.1f}, Median: {np.median(ranks):.1f}")
                    
                    # Count how many would be in top-K
                    top_10_count = sum(1 for r in ranks if r <= 10)
                    top_20_count = sum(1 for r in ranks if r <= 20)
                    top_50_count = sum(1 for r in ranks if r <= 50)
                    
                    print(f"  Previous purchases that would rank: Top-10: {top_10_count}/{len(ranks)} ({top_10_count/len(ranks)*100:.1f}%), Top-20: {top_20_count}/{len(ranks)} ({top_20_count/len(ranks)*100:.1f}%), Top-50: {top_50_count}/{len(ranks)} ({top_50_count/len(ranks)*100:.1f}%)")
        
        return recommendations
    
    def print_final_exclusion_summary(self):
        """Print final summary of purchase exclusion statistics"""
        if not self.exclude_prev_purchases:
            return
            
        print(f"\n" + "="*60)
        print(f"FINAL PURCHASE EXCLUSION SUMMARY")
        print(f"="*60)
        
        if self.total_recommendations_requested > 0:
            avg_exclusions = self.total_purchase_exclusions / self.total_recommendations_requested
            print(f"Total exclusions: {self.total_purchase_exclusions} across {self.total_recommendations_requested} users (avg: {avg_exclusions:.1f}/user)")
        
        if self.test_users_processed > 0:
            test_avg_exclusions = self.test_user_purchase_exclusions / self.test_users_processed
            print(f"Test user exclusions: {self.test_user_purchase_exclusions} across {self.test_users_processed} test users (avg: {test_avg_exclusions:.1f}/user)")
            
            if self.excluded_purchase_ranks:
                import numpy as np
                ranks = np.array(self.excluded_purchase_ranks)
                
                print(f"\nExcluded Purchase Ranking Analysis:")
                print(f"  Total excluded purchases tracked: {len(ranks)}")
                print(f"  Rank statistics - Min: {ranks.min()}, Max: {ranks.max()}, Mean: {ranks.mean():.1f}, Median: {np.median(ranks):.1f}")
                
                # Percentile analysis
                percentiles = [25, 50, 75, 90, 95, 99]
                percentile_values = np.percentile(ranks, percentiles)
                print(f"  Rank percentiles: " + ", ".join([f"P{p}: {v:.0f}" for p, v in zip(percentiles, percentile_values)]))
                
                # Top-K analysis
                top_k_analysis = [(10, "Top-10"), (20, "Top-20"), (50, "Top-50"), (100, "Top-100")]
                print(f"  Previous purchases that would have ranked in:")
                for k, label in top_k_analysis:
                    count = sum(1 for r in ranks if r <= k)
                    percentage = count/len(ranks)*100 if len(ranks) > 0 else 0
                    print(f"    {label}: {count}/{len(ranks)} ({percentage:.1f}%)")
                
                # Insight analysis
                very_high_rank = sum(1 for r in ranks if r <= 5)
                high_rank = sum(1 for r in ranks if r <= 20)
                low_rank = sum(1 for r in ranks if r > 100)
                
                print(f"\n  Model behavior insights:")
                if very_high_rank / len(ranks) > 0.3:
                    print(f"    ⚠️  High potential overfitting: {very_high_rank/len(ranks)*100:.1f}% of excluded purchases would rank in top-5")
                elif high_rank / len(ranks) > 0.5:
                    print(f"    ⚠️  Moderate overfitting: {high_rank/len(ranks)*100:.1f}% of excluded purchases would rank in top-20")
                else:
                    print(f"    ✅ Good generalization: Only {high_rank/len(ranks)*100:.1f}% of excluded purchases would rank in top-20")
                
                if low_rank / len(ranks) > 0.5:
                    print(f"    ✅ Good diversity: {low_rank/len(ranks)*100:.1f}% of excluded purchases rank below 100")
        
        print(f"="*60)
    
    def _create_batch(self, user_data: Dict) -> Dict:
        """Create a batch for the model"""
        
        # Get package features for short-term and long-term packages
        short_term_features = []
        for pkg_id in user_data['short_term_packages']:
            features = self.package_processor.get_package_features(str(pkg_id))
            if features:
                short_term_features.append(features)
        
        long_term_features = []
        for pkg_id in user_data['long_term_packages']:
            features = self.package_processor.get_package_features(str(pkg_id))
            if features:
                long_term_features.append(features)
        
        # Create batch structure similar to TravelPackageDataset
        batch = {
            'user_id': torch.tensor([user_data['user_idx']], dtype=torch.long),
            'short_term': self._create_sequence_batch(short_term_features, user_data['short_term_events']),
            'long_term': self._create_sequence_batch(long_term_features, user_data['long_term_events']),
            'purchased': self._create_single_item_batch({})  # Empty for inference
        }
        
        return batch
    
    def _create_sequence_batch(self, features_list: List[Dict], event_types: List[str]) -> Dict:
        """Create batch data for a sequence of packages"""
        # Get actual embedding dimension from the model config
        embedding_dim = self.model_info['config']['title_embedding_dim']
        
        if not features_list:
            # Return empty sequence
            return {
                'package_ids': torch.zeros(1, 1, dtype=torch.long),
                'title_embeddings': torch.zeros(1, 1, embedding_dim),
                'coordinates': torch.zeros(1, 1, 2),
                'country_ids': torch.zeros(1, 1, dtype=torch.long),
                'category_ids': torch.zeros(1, 1, dtype=torch.long),
                'theme_ids': torch.zeros(1, 1, dtype=torch.long),
                'event_types': torch.zeros(1, 1, dtype=torch.long),
                'prices': torch.zeros(1, 1, dtype=torch.float),
            }
        
        seq_len = len(features_list)
        
        # Initialize tensors
        package_ids = torch.zeros(1, seq_len, dtype=torch.long)
        title_embeddings = torch.zeros(1, seq_len, embedding_dim)
        coordinates = torch.zeros(1, seq_len, 2)
        country_ids = torch.zeros(1, seq_len, dtype=torch.long)
        category_ids = torch.zeros(1, seq_len, dtype=torch.long)
        theme_ids = torch.zeros(1, seq_len, dtype=torch.long)
        event_types_tensor = torch.zeros(1, seq_len, dtype=torch.long)
        prices = torch.zeros(1, seq_len, dtype=torch.float)
        
        # Fill tensors
        for i, (features, event_type) in enumerate(zip(features_list, event_types)):
            package_ids[0, i] = self.package_to_idx.get(features['main_id'], 0)
            
            if features['title_embedding'] is not None:
                # Handle dimension mismatch - truncate or pad as needed
                embedding = features['title_embedding']
                if len(embedding) > embedding_dim:
                    title_embeddings[0, i] = torch.tensor(embedding[:embedding_dim])
                elif len(embedding) < embedding_dim:
                    # Pad with zeros
                    padded = torch.zeros(embedding_dim)
                    padded[:len(embedding)] = torch.tensor(embedding)
                    title_embeddings[0, i] = padded
                else:
                    title_embeddings[0, i] = torch.tensor(embedding)
            
            if features['latitude'] is not None and features['longitude'] is not None:
                coordinates[0, i] = torch.tensor([features['latitude'], features['longitude']])
            
            country_ids[0, i] = features['country_idx']
            category_ids[0, i] = features['category_idx']
            theme_ids[0, i] = features['theme_idx']
            prices[0, i] = features['price']
            
            # Convert event type to index
            event_idx = self.session_processor.event_to_idx.get(event_type, 0)
            event_types_tensor[0, i] = event_idx
        
        return {
            'package_ids': package_ids,
            'title_embeddings': title_embeddings,
            'coordinates': coordinates,
            'country_ids': country_ids,
            'category_ids': category_ids,
            'theme_ids': theme_ids,
            'event_types': event_types_tensor,
            'prices': prices,
        }
    
    def _create_single_item_batch(self, features: Dict) -> Dict:
        """Create batch data for a single item (used for purchased item)"""
        # Get actual embedding dimension from the model config
        embedding_dim = self.model_info['config']['title_embedding_dim']
        
        return {
            'package_ids': torch.zeros(1, dtype=torch.long),
            'title_embeddings': torch.zeros(1, embedding_dim),
            'coordinates': torch.zeros(1, 2),
            'country_ids': torch.zeros(1, dtype=torch.long),
            'category_ids': torch.zeros(1, dtype=torch.long),
            'theme_ids': torch.zeros(1, dtype=torch.long),
            'prices': torch.zeros(1, dtype=torch.float),
        }


def main():
    """Test the recommender with availability and purchase history filtering"""
    
    # Initialize recommender
    recommender = NATRRecommender('model_info.json')
    
    # Test with a sample user
    test_user_id = "sample_user_123"  # Replace with actual user ID
    
    print(f"\n=== Testing Recommendations for User: {test_user_id} ===")
    
    # Test 1: Basic recommendations (without filtering)
    print("\n1. Basic recommendations (no filtering):")
    basic_recommendations = recommender.recommend(test_user_id, top_k=10, exclude_purchased=False)
    
    if basic_recommendations:
        print("Top 10 Basic Recommendations:")
        for rec in basic_recommendations:
            print(f"{rec['rank']}. {rec['title']} ({rec['city']}, {rec['country']})")
            print(f"   Score: {rec['score']:.4f}, Theme: {rec['theme']}, Price: €{rec['price']}")
    else:
        print("No basic recommendations found")
    
    # Test 2: Recommendations with purchase history filtering for this specific user
    print(f"\n2. Recommendations excluding previously purchased packages (user-specific filtering):")
    
    # Get user's purchase history
    purchased_packages = recommender.get_user_purchase_history(test_user_id)
    print(f"User {test_user_id} has purchased {len(purchased_packages)} packages previously")
    if purchased_packages:
        print(f"Previously purchased: {list(purchased_packages)[:5]}...")  # Show first 5
    
    filtered_recommendations = recommender.recommend(test_user_id, top_k=10, exclude_purchased=True)
    
    if filtered_recommendations:
        print("Top 10 Filtered Recommendations (new packages fill in automatically):")
        for rec in filtered_recommendations:
            print(f"{rec['rank']}. {rec['title']} ({rec['city']}, {rec['country']})")
            print(f"   Score: {rec['score']:.4f}, Theme: {rec['theme']}, Price: €{rec['price']}")
        
        # Verify none of the recommended packages were previously purchased
        recommended_ids = {rec['package_id'] for rec in filtered_recommendations}
        overlap = recommended_ids & purchased_packages
        if overlap:
            print(f"⚠️  WARNING: Found overlap with purchased packages: {overlap}")
        else:
            print("✅ Confirmed: No recommended packages were previously purchased by this user")
    else:
        print("No filtered recommendations found")
    
    # Test 3: Recommendations with availability filtering (simulate limited availability)
    print(f"\n3. Recommendations with availability filtering:")
    
    # Simulate limited availability by taking only a subset of valid packages
    all_packages = list(recommender.valid_packages)
    if len(all_packages) > 100:
        # Simulate that only 70% of packages are available
        import random
        random.seed(42)  # For reproducible results
        available_count = int(len(all_packages) * 0.7)
        simulated_available = set(random.sample(all_packages, available_count))
        print(f"Simulating {len(simulated_available)} available packages out of {len(all_packages)} total")
        
        availability_filtered_recommendations = recommender.recommend(
            test_user_id, 
            top_k=10, 
            available_packages=simulated_available,
            exclude_purchased=True
        )
        
        if availability_filtered_recommendations:
            print("Top 10 Availability-Filtered Recommendations:")
            for rec in availability_filtered_recommendations:
                print(f"{rec['rank']}. {rec['title']} ({rec['city']}, {rec['country']})")
                print(f"   Score: {rec['score']:.4f}, Theme: {rec['theme']}, Price: €{rec['price']}")
        else:
            print("No availability-filtered recommendations found")
    else:
        print("Not enough packages to simulate availability filtering")
    
    # Test 4: Helper method testing
    print(f"\n4. Testing helper methods:")
    
    # Test availability filtering
    sample_packages = list(recommender.valid_packages)[:20] if len(recommender.valid_packages) >= 20 else list(recommender.valid_packages)
    simulated_available_subset = set(sample_packages[:10])  # Only first 10 are available
    
    available_filtered = recommender.filter_available_packages(sample_packages, simulated_available_subset)
    print(f"Availability filter test: {len(sample_packages)} → {len(available_filtered)} packages")
    
    # Test purchase history filtering
    unpurchased_filtered = recommender.filter_unpurchased_packages(sample_packages, test_user_id)
    print(f"Purchase history filter test: {len(sample_packages)} → {len(unpurchased_filtered)} packages")
    
    print(f"\n=== Recommendation Testing Complete ===")


def demonstrate_user_specific_filtering():
    """Demonstrate how purchase history filtering works differently for different users"""
    
    print("\n=== User-Specific Purchase History Filtering Demo ===")
    
    # Initialize recommender
    recommender = NATRRecommender('model_info.json')
    
    # Get a list of users with different purchase histories
    users_with_purchases = [user_id for user_id, purchases in recommender._purchase_history_cache.items() if len(purchases) > 0]
    
    if len(users_with_purchases) >= 2:
        # Compare two different users
        user1 = users_with_purchases[0]
        user2 = users_with_purchases[1]
        
        print(f"Comparing recommendations for two different users:")
        print(f"User 1: {user1}")
        print(f"User 2: {user2}")
        
        for user_id in [user1, user2]:
            print(f"\n--- Recommendations for {user_id} ---")
            
            # Get purchase history
            purchased = recommender.get_user_purchase_history(user_id)
            print(f"Previously purchased: {len(purchased)} packages")
            
            # Get recommendations without filtering
            basic_recs = recommender.recommend(user_id, top_k=5, exclude_purchased=False)
            print(f"Basic recommendations (top 5):")
            for rec in basic_recs[:3]:  # Show only first 3
                print(f"  {rec['rank']}. {rec['title']} (Score: {rec['score']:.3f})")
            
            # Get recommendations with purchase filtering
            filtered_recs = recommender.recommend(user_id, top_k=5, exclude_purchased=True)
            print(f"Filtered recommendations (excluding purchased):")
            for rec in filtered_recs[:3]:  # Show only first 3
                print(f"  {rec['rank']}. {rec['title']} (Score: {rec['score']:.3f})")
            
            # Check if any recommended packages were purchased
            if basic_recs and filtered_recs:
                basic_ids = {rec['package_id'] for rec in basic_recs}
                filtered_ids = {rec['package_id'] for rec in filtered_recs}
                excluded_from_top5 = basic_ids & purchased
                
                if excluded_from_top5:
                    print(f"  ✅ Excluded {len(excluded_from_top5)} purchased packages from top 5")
                    print(f"  ✅ New packages automatically filled the top 5 slots")
                else:
                    print(f"  ℹ️  No purchased packages were in the top 5 for this user")
    else:
        print("Not enough users with purchase history for demonstration")


def demonstrate_filtering_workflow():
    """Demonstrate a complete workflow with filtering for production use"""
    
    print("\n=== Production Workflow Demonstration ===")
    
    # Initialize recommender
    recommender = NATRRecommender('model_info.json')
    
    # Example user
    user_id = "example_user_456"  # Replace with actual user ID
    
    # Step 1: Get available packages from your inventory system
    # In production, this would come from your package availability API
    print("Step 1: Checking package availability...")
    # Simulate getting available packages from inventory
    all_packages = list(recommender.valid_packages)
    # In production: available_packages = get_available_packages_from_inventory()
    available_packages = set(all_packages[:int(len(all_packages) * 0.8)])  # Simulate 80% availability
    print(f"Found {len(available_packages)} available packages out of {len(all_packages)} total")
    
    # Step 2: Get recommendations with all filtering applied
    print("Step 2: Generating filtered recommendations...")
    recommendations = recommender.recommend(
        user_id=user_id,
        top_k=10,
        available_packages=available_packages,
        exclude_purchased=True
    )
    
    # Step 3: Present results
    print("Step 3: Final recommendations:")
    if recommendations:
        print(f"Successfully generated {len(recommendations)} recommendations:")
        for rec in recommendations:
            print(f"  {rec['rank']}. {rec['title']} - €{rec['price']:.2f}")
            print(f"     Location: {rec['city']}, {rec['country']}")
            print(f"     Theme: {rec['theme']}, Score: {rec['score']:.4f}")
            print()
    else:
        print("  No recommendations available after filtering")
    
    return recommendations


if __name__ == "__main__":
    main()