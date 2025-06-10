#!/usr/bin/env python3
"""
Generate Batch Recommendations for All Users - Production Script

This script generates recommendations for all users in the dataset, suitable for
daily production runs. Outputs recommendations in various formats for downstream
consumption.

Usage:
    # Generate recommendations for all users
    python scripts/generate_batch_recommendations.py --model-info output/model_info_2months/model_info.json
    
    # With availability filtering
    python scripts/generate_batch_recommendations.py --model-info output/model_info_2months/model_info.json \
        --availability-data data/availability.parquet --min-available-days 5
    
    # Limit to subset of users for testing
    python scripts/generate_batch_recommendations.py --model-info output/model_info_2months/model_info.json \
        --max-users 1000 --output-format parquet
"""

import argparse
import json
import sys
import os
import pandas as pd
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Set
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from utils.recommendations import NATRRecommender


class BatchRecommendationGenerator:
    """
    Generate recommendations for all users in batch for production deployment
    """
    
    def __init__(self, model_info_path: str, availability_data_path: Optional[str] = None):
        """
        Initialize batch recommendation generator
        
        Args:
            model_info_path: Path to model_info.json file
            availability_data_path: Optional path to availability data
        """
        self.model_info_path = model_info_path
        self.availability_data_path = availability_data_path
        
        # Load model info
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        print(f"Generating batch recommendations with model: {os.path.basename(model_info_path)}")
        print(f"Training strategy: {self.model_info.get('training_strategy', 'standard')}")
        
        # Initialize recommender
        print("\nInitializing recommender...")
        self.recommender = NATRRecommender(
            model_info_path=model_info_path,
            availability_data_path=availability_data_path
        )
        
        print("Batch recommendation generator ready!")
    
    def get_all_users(self, max_users: Optional[int] = None) -> List[str]:
        """
        Get all users from the dataset
        
        Args:
            max_users: Optional limit on number of users (for testing)
            
        Returns:
            List of user IDs
        """
        all_users = list(self.recommender.user_to_idx.keys())
        
        # Filter out users without purchase history if needed
        users_with_history = []
        for user_id in all_users:
            if user_id in self.recommender._purchase_history_cache:
                users_with_history.append(user_id)
        
        print(f"Total users in dataset: {len(all_users):,}")
        print(f"Users with purchase history: {len(users_with_history):,}")
        
        # Use users with history for better recommendations
        target_users = users_with_history if users_with_history else all_users
        
        # Limit if requested
        if max_users and max_users < len(target_users):
            # Take a representative sample
            import random
            random.seed(42)
            target_users = random.sample(target_users, max_users)
            print(f"Limited to {max_users:,} users for processing")
        
        return target_users
    
    def generate_recommendations_for_users(self, 
                                         user_ids: List[str],
                                         top_k: int = 20,
                                         min_available_days: int = 5,
                                         exclude_purchased: bool = True,
                                         batch_size: int = 100) -> List[Dict]:
        """
        Generate recommendations for a list of users
        
        Args:
            user_ids: List of user IDs to generate recommendations for
            top_k: Number of recommendations per user
            min_available_days: Minimum availability days for packages
            exclude_purchased: Whether to exclude previously purchased items
            batch_size: Number of users to process in each progress update
            
        Returns:
            List of recommendation dictionaries
        """
        print(f"\nGenerating {top_k} recommendations for {len(user_ids):,} users...")
        if min_available_days > 0:
            print(f"Filtering for packages with at least {min_available_days} available days")
        if exclude_purchased:
            print("Excluding previously purchased packages")
        
        all_recommendations = []
        failed_users = []
        
        start_time = time.time()
        
        # Process users in batches for progress tracking
        for i in tqdm(range(0, len(user_ids), batch_size), desc="Processing user batches"):
            batch_users = user_ids[i:i + batch_size]
            
            for user_id in batch_users:
                try:
                    # Generate recommendations for this user
                    recommendations = self.recommender.recommend(
                        user_id=user_id,
                        top_k=top_k,
                        available_packages=None,  # Use availability_data filtering instead
                        min_available_days=min_available_days,
                        exclude_purchased=exclude_purchased
                    )
                    
                    if recommendations:
                        # Add user ID and metadata to each recommendation
                        user_recommendations = {
                            'user_id': user_id,
                            'recommendations': recommendations,
                            'num_recommendations': len(recommendations),
                            'generated_at': datetime.now().isoformat()
                        }
                        all_recommendations.append(user_recommendations)
                    else:
                        failed_users.append(user_id)
                
                except Exception as e:
                    print(f"Error generating recommendations for user {user_id}: {str(e)}")
                    failed_users.append(user_id)
            
            # Memory cleanup for MPS/CUDA
            if i % (batch_size * 10) == 0:  # Every 1000 users
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        
        elapsed_time = time.time() - start_time
        successful_users = len(all_recommendations)
        
        print(f"\nRecommendation generation complete!")
        print(f"Successful: {successful_users:,}/{len(user_ids):,} users ({successful_users/len(user_ids)*100:.1f}%)")
        print(f"Failed: {len(failed_users):,} users")
        print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/len(user_ids):.3f}s per user)")
        
        if failed_users:
            print(f"Failed users (first 10): {failed_users[:10]}")
        
        return all_recommendations
    
    def save_recommendations(self, 
                           recommendations: List[Dict], 
                           output_path: str, 
                           format: str = 'parquet') -> str:
        """
        Save recommendations to file in specified format
        
        Args:
            recommendations: List of recommendation dictionaries
            output_path: Base output path (extension will be added based on format)
            format: Output format ('parquet', 'json', 'csv')
            
        Returns:
            Full path to saved file
        """
        if not recommendations:
            print("No recommendations to save")
            return None
        
        # Prepare filename with timestamp and format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"batch_recommendations_{timestamp}"
        
        if format == 'parquet':
            # Flatten for parquet format
            flat_data = []
            for user_rec in recommendations:
                user_id = user_rec['user_id']
                generated_at = user_rec['generated_at']
                
                for rank, rec in enumerate(user_rec['recommendations'], 1):
                    flat_data.append({
                        'user_id': user_id,
                        'rank': rank,
                        'package_id': rec['package_id'],
                        'score': rec['score'],
                        'package_title': rec.get('title', ''),
                        'package_category': rec.get('category', ''),
                        'package_country': rec.get('country', ''),
                        'package_price': rec.get('price', 0.0),
                        'available_days': rec.get('available_days', 0),
                        'generated_at': generated_at
                    })
            
            df = pd.DataFrame(flat_data)
            output_file = f"{output_path}/{base_name}.parquet"
            df.to_parquet(output_file, index=False)
            
            print(f"Saved {len(flat_data):,} recommendations to parquet: {output_file}")
            print(f"Parquet file size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            
        elif format == 'json':
            output_file = f"{output_path}/{base_name}.json"
            
            # Add summary metadata
            summary_data = {
                'metadata': {
                    'total_users': len(recommendations),
                    'total_recommendations': sum(len(r['recommendations']) for r in recommendations),
                    'model_info_path': self.model_info_path,
                    'availability_data_path': self.availability_data_path,
                    'generated_at': datetime.now().isoformat()
                },
                'recommendations': recommendations
            }
            
            with open(output_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            print(f"Saved recommendations to JSON: {output_file}")
            print(f"JSON file size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            
        elif format == 'csv':
            # Similar to parquet but save as CSV
            flat_data = []
            for user_rec in recommendations:
                user_id = user_rec['user_id']
                generated_at = user_rec['generated_at']
                
                for rank, rec in enumerate(user_rec['recommendations'], 1):
                    flat_data.append({
                        'user_id': user_id,
                        'rank': rank,
                        'package_id': rec['package_id'],
                        'score': rec['score'],
                        'package_title': rec.get('title', ''),
                        'package_category': rec.get('category', ''),
                        'package_country': rec.get('country', ''),
                        'package_price': rec.get('price', 0.0),
                        'available_days': rec.get('available_days', 0),
                        'generated_at': generated_at
                    })
            
            df = pd.DataFrame(flat_data)
            output_file = f"{output_path}/{base_name}.csv"
            df.to_csv(output_file, index=False)
            
            print(f"Saved {len(flat_data):,} recommendations to CSV: {output_file}")
            print(f"CSV file size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_file
    
    def generate_batch_recommendations(self,
                                     top_k: int = 20,
                                     max_users: Optional[int] = None,
                                     min_available_days: int = 5,
                                     exclude_purchased: bool = True,
                                     output_dir: str = 'output/recommendations',
                                     output_format: str = 'parquet') -> str:
        """
        Complete batch recommendation generation pipeline
        
        Args:
            top_k: Number of recommendations per user
            max_users: Optional limit on number of users
            min_available_days: Minimum availability requirement
            exclude_purchased: Whether to exclude purchased items
            output_dir: Directory to save recommendations
            output_format: Output format ('parquet', 'json', 'csv')
            
        Returns:
            Path to saved recommendations file
        """
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all users
        user_ids = self.get_all_users(max_users)
        
        if not user_ids:
            print("No users found to generate recommendations for")
            return None
        
        # Generate recommendations
        recommendations = self.generate_recommendations_for_users(
            user_ids=user_ids,
            top_k=top_k,
            min_available_days=min_available_days,
            exclude_purchased=exclude_purchased
        )
        
        if not recommendations:
            print("No recommendations generated")
            return None
        
        # Save recommendations
        output_file = self.save_recommendations(
            recommendations=recommendations,
            output_path=output_dir,
            format=output_format
        )
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Batch recommendation generation complete!")
        print(f"Total processing time: {total_time:.1f}s")
        print(f"Output saved to: {output_file}")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate batch recommendations for all users (production script)'
    )
    
    # Required arguments
    parser.add_argument('--model-info', type=str, required=True,
                        help='Path to model_info.json file')
    
    # Optional data arguments
    parser.add_argument('--availability-data', type=str,
                        help='Path to availability.parquet file')
    parser.add_argument('--min-available-days', type=int, default=5,
                        help='Minimum number of available days required (default: 5)')
    
    # Generation parameters
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of recommendations per user (default: 20)')
    parser.add_argument('--max-users', type=int,
                        help='Maximum number of users to process (for testing)')
    parser.add_argument('--no-exclude-purchased', action='store_true',
                        help='Do NOT exclude previously purchased packages')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output/recommendations',
                        help='Output directory for recommendations (default: output/recommendations)')
    parser.add_argument('--output-format', choices=['parquet', 'json', 'csv'], 
                        default='parquet',
                        help='Output format (default: parquet)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_info):
        print(f"Error: Model info file not found: {args.model_info}")
        return 1
    
    if args.availability_data and not os.path.exists(args.availability_data):
        print(f"Error: Availability data file not found: {args.availability_data}")
        return 1
    
    try:
        # Initialize generator
        generator = BatchRecommendationGenerator(
            model_info_path=args.model_info,
            availability_data_path=args.availability_data
        )
        
        # Generate batch recommendations
        output_file = generator.generate_batch_recommendations(
            top_k=args.top_k,
            max_users=args.max_users,
            min_available_days=args.min_available_days,
            exclude_purchased=not args.no_exclude_purchased,
            output_dir=args.output_dir,
            output_format=args.output_format
        )
        
        if output_file:
            print(f"\n🎉 Success! Recommendations saved to: {output_file}")
            return 0
        else:
            print("\n❌ Failed to generate recommendations")
            return 1
        
    except Exception as e:
        print(f"\nError during batch recommendation generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())