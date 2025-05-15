"""
NATR inference script for making recommendations
"""

import torch
import json
import numpy as np
import sys
import os
from typing import List, Dict, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.natr import NATR, NATRConfig
from utils.session_processor import SessionProcessor
from utils.package_processor import PackageProcessor


class NATRRecommender:
    """NATR model for inference"""
    
    def __init__(self, model_info_path='model_info.json'):
        """Initialize recommender from saved model"""
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Initialize processors
        self.package_processor = PackageProcessor(
            data_path=self.model_info['package_data_path'],
            cache_dir='data/cache',
            load_coordinates=True,
            load_embeddings=True
        )
        
        self.session_processor = SessionProcessor(
            data_path=self.model_info['event_data_path'],
            cache_dir='data/cache'
        )
        
        # Load data
        print("Loading data...")
        self.package_processor.load_data()
        self.session_processor.load_data()
        
        # Create mappings
        self.package_processor.create_mappings()
        self.session_processor.create_mappings()
        
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
        
        # Valid packages
        self.valid_packages = set(self.model_info['valid_packages'])
        
        print("Recommender ready!")
    
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
    
    def recommend(self, user_id: str, top_k: int = 10) -> List[Dict]:
        """Get recommendations for a user"""
        
        # Prepare user data
        user_data = self.prepare_user_data(user_id)
        if user_data is None:
            return []
        
        # Create batch
        batch = self._create_batch(user_data)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(batch)
            predictions = outputs['predictions']
        
        # Get top-k
        scores, indices = torch.topk(predictions[0], top_k)
        
        # Convert to package info
        recommendations = []
        for i in range(top_k):
            pkg_idx = indices[i].item()
            pkg_id = self.idx_to_package.get(pkg_idx)
            
            if pkg_id and str(pkg_id) in self.valid_packages:
                pkg_info = self.package_processor.get_package_features(str(pkg_id))
                
                recommendations.append({
                    'rank': i + 1,
                    'package_id': pkg_id,
                    'score': scores[i].item(),
                    'title': pkg_info.get('title', ''),
                    'city': pkg_info.get('city', ''),
                    'country': pkg_info.get('country', ''),
                    'theme': pkg_info.get('theme', ''),
                    'category': pkg_info.get('category', ''),
                    'price': pkg_info.get('price', 0)
                })
        
        return recommendations
    
    def _create_batch(self, user_data: Dict) -> Dict:
        """Create a batch for the model"""
        
        # Similar to TravelPackageDataset but for single user
        # (Implementation details omitted for brevity)
        # This would create the proper batch structure with all features
        
        pass


def main():
    """Test the recommender"""
    
    # Initialize recommender
    recommender = NATRRecommender('model_info.json')
    
    # Test with a sample user
    test_user_id = "sample_user_123"  # Replace with actual user ID
    
    print(f"\nGetting recommendations for user: {test_user_id}")
    recommendations = recommender.recommend(test_user_id, top_k=10)
    
    if recommendations:
        print("\nTop 10 Recommendations:")
        for rec in recommendations:
            print(f"{rec['rank']}. {rec['title']} ({rec['city']}, {rec['country']})")
            print(f"   Score: {rec['score']:.4f}, Theme: {rec['theme']}, Price: ${rec['price']}")
    else:
        print("No recommendations found")


if __name__ == "__main__":
    main()