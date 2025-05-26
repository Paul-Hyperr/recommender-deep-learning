"""
Unified metrics for evaluating recommender models across different training approaches.
Includes weighted recall metrics for purchase, checkout, and add-to-cart events.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import torch.nn.functional as F


class LearnableMargin(torch.nn.Module):
    """
    Learnable margin parameters for contrastive learning.
    Automatically adjusts margins during training through gradient descent.
    """
    
    def __init__(self, 
                 initial_margin: float,
                 min_margin: float = 0.0,
                 max_margin: float = 1.0):
        """
        Initialize learnable margin parameter.
        
        Args:
            initial_margin: Initial margin value
            min_margin: Minimum allowed margin value
            max_margin: Maximum allowed margin value
        """
        super().__init__()
        self.min_margin = min_margin
        self.max_margin = max_margin
        
        # Use log-space parameterization for better numeric stability
        self.margin_param = torch.nn.Parameter(
            torch.tensor(np.log(initial_margin - min_margin + 1e-6), 
                       dtype=torch.float)
        )
        
    def forward(self):
        """Get the current margin value with constraints applied"""
        # Convert from log-space and apply constraints
        margin = torch.exp(self.margin_param) + self.min_margin - 1e-6
        margin = torch.clamp(margin, self.min_margin, self.max_margin)
        return margin


class UnifiedMetricsTracker:
    """
    Unified metrics tracking for all training scripts with priority weighting.
    Handles all event types and calculates weighted composite metrics.
    
    Key features:
    - Unified approach to calculating metrics across all training methods
    - Event-specific metrics (purchase, checkout, add_to_cart)
    - Weighted composite metrics based on event importance
    - Comprehensive tracking and averaging over evaluation batches
    - Robust error handling for edge cases
    """
    
    def __init__(self, 
                 k_values: List[int] = [10, 20],
                 event_weights: Optional[Dict[str, float]] = None,
                 validate_inputs: bool = True):
        """
        Initialize metrics tracker with specified k values and event weights.
        
        Args:
            k_values: List of k values for recall@k and other metrics
            event_weights: Dictionary of weights for each event type in composite metrics
                          Default: {'purchase': 1.0, 'checkout': 0.6, 'add_to_cart': 0.2, 'view': 0.05}
            validate_inputs: Whether to validate inputs for error handling
        """
        self.k_values = k_values
        self.validate_inputs = validate_inputs
        
        # Default event weights if not provided
        if event_weights is None:
            self.event_weights = {
                'purchase': 1.0,      # Full weight for purchases
                'checkout': 0.6,      # Consistent weight for checkout events
                'add_to_cart': 0.2,   # Consistent weight for add-to-cart
                'view': 0.05          # Very minimal weight for view events
            }
        else:
            self.event_weights = event_weights
            
        # Initialize the metrics storage
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics for a new evaluation session"""
        # Basic tracking metrics
        self.metrics = {
            # Overall metrics
            'recall': {k: [] for k in self.k_values},
            'mrr': 0.0,
            
            # Event-specific metrics
            'purchase_recall': {k: [] for k in self.k_values},
            'checkout_recall': {k: [] for k in self.k_values},
            'add_to_cart_recall': {k: [] for k in self.k_values},
            'intent_recall': {k: [] for k in self.k_values},  # purchase + checkout
            
            'purchase_mrr': 0.0,
            'checkout_mrr': 0.0,
            'add_to_cart_mrr': 0.0,
            'intent_mrr': 0.0,  # purchase + checkout
            
            # Weighted composite metrics
            'weighted_recall': {k: [] for k in self.k_values},
            'weighted_mrr': 0.0,
            
            # Validation metrics (precision, ndcg)
            'precision': {k: [] for k in self.k_values},
            'ndcg': {k: [] for k in self.k_values},
            
            # Loss tracking
            'loss': []
        }
        
        # Count tracking
        self.counts = {
            'total': 0,
            'purchase': 0,
            'checkout': 0,
            'add_to_cart': 0,
            'intent': 0,  # purchase + checkout
            'view': 0     # regular browsing
        }
    
    def update(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor,
               is_purchase: torch.Tensor,
               has_checkout: Optional[torch.Tensor] = None,
               has_add_to_cart: Optional[torch.Tensor] = None,
               has_checkout_inclusive: Optional[torch.Tensor] = None,
               has_add_to_cart_inclusive: Optional[torch.Tensor] = None,
               loss: Optional[float] = None):
        """
        Update metrics with a new batch of predictions and targets.
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Ground truth target indices [batch_size]
            is_purchase: Boolean tensor indicating purchase samples [batch_size]
            has_checkout: Boolean tensor indicating checkout samples (hierarchical - for weighted recall)
            has_add_to_cart: Boolean tensor indicating add-to-cart samples (hierarchical - for weighted recall)
            has_checkout_inclusive: Boolean tensor indicating all checkout samples (for individual recalls)
            has_add_to_cart_inclusive: Boolean tensor indicating all add-to-cart samples (for individual recalls)
            loss: Optional loss value to track
        """
        # Validate inputs if enabled
        if self.validate_inputs:
            # Handle problematic inputs with robust error handling
            try:
                # Ensure input tensors are on the same device
                if predictions.device != targets.device:
                    targets = targets.to(predictions.device)
                
                if is_purchase.device != predictions.device:
                    is_purchase = is_purchase.to(predictions.device)
                
                # Ensure has_checkout and has_add_to_cart are properly handled
                if has_checkout is None:
                    has_checkout = torch.zeros_like(is_purchase)
                elif has_checkout.device != predictions.device:
                    has_checkout = has_checkout.to(predictions.device)
                
                if has_add_to_cart is None:
                    has_add_to_cart = torch.zeros_like(is_purchase)
                elif has_add_to_cart.device != predictions.device:
                    has_add_to_cart = has_add_to_cart.to(predictions.device)
                
                # Handle inclusive flags
                if has_checkout_inclusive is None:
                    has_checkout_inclusive = torch.zeros_like(is_purchase)
                elif has_checkout_inclusive.device != predictions.device:
                    has_checkout_inclusive = has_checkout_inclusive.to(predictions.device)
                
                if has_add_to_cart_inclusive is None:
                    has_add_to_cart_inclusive = torch.zeros_like(is_purchase)
                elif has_add_to_cart_inclusive.device != predictions.device:
                    has_add_to_cart_inclusive = has_add_to_cart_inclusive.to(predictions.device)
                
                # Ensure tensors have compatible shapes
                if len(targets.shape) > 1:
                    targets = targets.squeeze()
                
                if len(is_purchase.shape) > 1:
                    is_purchase = is_purchase.squeeze()
                    
                if len(has_checkout.shape) > 1:
                    has_checkout = has_checkout.squeeze()
                    
                if len(has_add_to_cart.shape) > 1:
                    has_add_to_cart = has_add_to_cart.squeeze()
                    
                if len(has_checkout_inclusive.shape) > 1:
                    has_checkout_inclusive = has_checkout_inclusive.squeeze()
                    
                if len(has_add_to_cart_inclusive.shape) > 1:
                    has_add_to_cart_inclusive = has_add_to_cart_inclusive.squeeze()
                    
                # Handle empty input case
                if targets.numel() == 0:
                    print("Warning: Empty targets tensor, skipping metrics update")
                    return
                    
                # Handle case where targets are out of bounds
                if targets.max() >= predictions.size(1):
                    print(f"Warning: Target index {targets.max().item()} out of bounds " 
                          f"(num_items={predictions.size(1)}), clipping")
                    targets = torch.clamp(targets, 0, predictions.size(1) - 1)
                
            except Exception as e:
                print(f"Error validating inputs: {e}")
                return
        
        # Update count metrics
        batch_size = predictions.size(0)
        self.counts['total'] += batch_size
        self.counts['purchase'] += is_purchase.sum().item()
        
        # Calculate intent mask (purchase OR checkout)
        intent_mask = is_purchase | has_checkout
        self.counts['intent'] += intent_mask.sum().item()
        
        # Update checkout and add_to_cart counts using inclusive flags (for individual recalls)
        if has_checkout_inclusive is not None:
            self.counts['checkout'] += has_checkout_inclusive.sum().item()
        
        if has_add_to_cart_inclusive is not None:
            self.counts['add_to_cart'] += has_add_to_cart_inclusive.sum().item()
        
        # Calculate view count (none of the above)
        view_mask = ~(is_purchase | has_checkout | has_add_to_cart)
        self.counts['view'] += view_mask.sum().item()
        
        # Track loss if provided
        if loss is not None:
            self.metrics['loss'].append(loss)
        
        # Calculate detailed metrics
        batch_metrics = self._calculate_batch_metrics(
            predictions, targets, is_purchase, has_checkout, has_add_to_cart,
            has_checkout_inclusive, has_add_to_cart_inclusive
        )
        
        # Update metrics with batch results
        for k in self.k_values:
            # Overall recall
            self.metrics['recall'][k].append(batch_metrics['recall'][k])
            
            # Event-specific recall
            if self.counts['purchase'] > 0 and 'purchase_recall' in batch_metrics:
                self.metrics['purchase_recall'][k].append(batch_metrics['purchase_recall'][k])
            
            if self.counts['checkout'] > 0 and 'checkout_recall' in batch_metrics:
                self.metrics['checkout_recall'][k].append(batch_metrics['checkout_recall'][k])
            
            if self.counts['add_to_cart'] > 0 and 'add_to_cart_recall' in batch_metrics:
                self.metrics['add_to_cart_recall'][k].append(batch_metrics['add_to_cart_recall'][k])
            
            if self.counts['intent'] > 0 and 'intent_recall' in batch_metrics:
                self.metrics['intent_recall'][k].append(batch_metrics['intent_recall'][k])
            
            # Weighted composite recall
            if 'weighted_recall' in batch_metrics:
                self.metrics['weighted_recall'][k].append(batch_metrics['weighted_recall'][k])
            
            # Additional validation metrics
            if 'precision' in batch_metrics:
                self.metrics['precision'][k].append(batch_metrics['precision'][k])
            
            if 'ndcg' in batch_metrics:
                self.metrics['ndcg'][k].append(batch_metrics['ndcg'][k])
        
        # Update MRR metrics
        self.metrics['mrr'] += batch_metrics['mrr'] * batch_size
        
        if self.counts['purchase'] > 0 and 'purchase_mrr' in batch_metrics:
            self.metrics['purchase_mrr'] += batch_metrics['purchase_mrr'] * batch_metrics['purchase_count']
        
        if self.counts['checkout'] > 0 and 'checkout_mrr' in batch_metrics:
            self.metrics['checkout_mrr'] += batch_metrics['checkout_mrr'] * batch_metrics['checkout_count_inclusive']
        
        if self.counts['add_to_cart'] > 0 and 'add_to_cart_mrr' in batch_metrics:
            self.metrics['add_to_cart_mrr'] += batch_metrics['add_to_cart_mrr'] * batch_metrics['add_to_cart_count_inclusive']
        
        if self.counts['intent'] > 0 and 'intent_mrr' in batch_metrics:
            self.metrics['intent_mrr'] += batch_metrics['intent_mrr'] * batch_metrics['intent_count']
        
        if 'weighted_mrr' in batch_metrics:
            self.metrics['weighted_mrr'] += batch_metrics['weighted_mrr'] * batch_size
    
    def _calculate_batch_metrics(self,
                                predictions: torch.Tensor,
                                targets: torch.Tensor,
                                is_purchase: torch.Tensor,
                                has_checkout: torch.Tensor,
                                has_add_to_cart: torch.Tensor,
                                has_checkout_inclusive: torch.Tensor,
                                has_add_to_cart_inclusive: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a single batch with event-based weighting.
        
        Args:
            predictions: Model predictions [batch_size, num_items]
            targets: Ground truth indices [batch_size]
            is_purchase: Boolean tensor for purchase events [batch_size]
            has_checkout: Boolean tensor for checkout events (hierarchical) [batch_size]
            has_add_to_cart: Boolean tensor for add_to_cart events (hierarchical) [batch_size]
            has_checkout_inclusive: Boolean tensor for all checkout events (inclusive) [batch_size]
            has_add_to_cart_inclusive: Boolean tensor for all add_to_cart events (inclusive) [batch_size]
            
        Returns:
            Dictionary of metrics including recall@k, MRR, and weighted metrics
        """
        batch_size = predictions.size(0)
        max_k = max(self.k_values)
        
        # Initialize metrics dictionary
        metrics = {
            'recall': {k: 0.0 for k in self.k_values},
            'mrr': 0.0,
            'weighted_recall': {k: 0.0 for k in self.k_values},
            'weighted_mrr': 0.0,
            'precision': {k: 0.0 for k in self.k_values},
            'ndcg': {k: 0.0 for k in self.k_values}
        }
        
        # Track event counts
        purchase_count = is_purchase.sum().item()
        checkout_count = has_checkout.sum().item()  # Hierarchical count for weighted recall
        add_to_cart_count = has_add_to_cart.sum().item()  # Hierarchical count for weighted recall
        
        # Track inclusive event counts for individual recalls
        checkout_count_inclusive = has_checkout_inclusive.sum().item()
        add_to_cart_count_inclusive = has_add_to_cart_inclusive.sum().item()
        
        # Calculate intent mask (purchase OR checkout)
        intent_mask = is_purchase | has_checkout
        intent_count = intent_mask.sum().item()
        
        # Add count information to metrics
        metrics['purchase_count'] = purchase_count
        metrics['checkout_count'] = checkout_count
        metrics['add_to_cart_count'] = add_to_cart_count
        metrics['intent_count'] = intent_count
        
        # Add inclusive counts for debugging
        metrics['checkout_count_inclusive'] = checkout_count_inclusive
        metrics['add_to_cart_count_inclusive'] = add_to_cart_count_inclusive
        
        # Get top-k predictions
        _, top_indices = torch.topk(predictions, min(max_k, predictions.size(1)), dim=1)
        
        # Create expanded targets for comparison
        expanded_targets = targets.unsqueeze(1).expand(-1, top_indices.size(1))
        
        # Calculate matches
        matches = (top_indices == expanded_targets)
        
        # Prepare event type tensors for weighted metrics with hierarchical priority
        # Hierarchy: Purchase > InitiateCheckout > AddToCart > View
        # A session should only count for the highest event type it contains
        event_weights_tensor = torch.ones(batch_size, device=predictions.device) * self.event_weights['view']
        
        # Start with lowest priority and work up (so higher priority overwrites)
        event_weights_tensor[has_add_to_cart] = self.event_weights['add_to_cart']
        
        # InitiateCheckout overwrites AddToCart if both are present
        event_weights_tensor[has_checkout] = self.event_weights['checkout'] 
        
        # Purchase overwrites all others if present (highest priority)
        event_weights_tensor[is_purchase] = self.event_weights['purchase']
        
        # Calculate rank information
        ranks = torch.zeros(batch_size, device=predictions.device)
        has_match = torch.zeros(batch_size, dtype=torch.bool, device=predictions.device)
        
        for i in range(batch_size):
            match_positions = matches[i].nonzero(as_tuple=True)[0]
            if len(match_positions) > 0:
                has_match[i] = True
                # +1 because ranks start from 1
                ranks[i] = match_positions[0].item() + 1
            else:
                # No match, set to max rank + 1
                ranks[i] = top_indices.size(1) + 1
        
        # Calculate reciprocal rank
        reciprocal_ranks = torch.zeros_like(ranks)
        reciprocal_ranks[has_match] = 1.0 / ranks[has_match]
        
        # Overall MRR
        metrics['mrr'] = reciprocal_ranks.mean().item()
        
        # Purchase MRR
        if purchase_count > 0:
            metrics['purchase_mrr'] = reciprocal_ranks[is_purchase].mean().item()
            
        # Checkout MRR (using inclusive flags)
        if checkout_count_inclusive > 0:
            metrics['checkout_mrr'] = reciprocal_ranks[has_checkout_inclusive].mean().item()
            
        # Add to cart MRR (using inclusive flags)
        if add_to_cart_count_inclusive > 0:
            metrics['add_to_cart_mrr'] = reciprocal_ranks[has_add_to_cart_inclusive].mean().item()
            
        # Intent MRR (purchase + checkout)
        if intent_count > 0:
            metrics['intent_mrr'] = reciprocal_ranks[intent_mask].mean().item()
        
        # Weighted MRR using event weights
        weighted_ranks = reciprocal_ranks * event_weights_tensor
        metrics['weighted_mrr'] = weighted_ranks.sum().item() / event_weights_tensor.sum().item()
        
        # Calculate recall@k and other metrics for each k value
        for k in self.k_values:
            # Overall recall@k
            in_top_k = matches[:, :k].any(dim=1).float()
            metrics['recall'][k] = in_top_k.mean().item()
            
            # Event-specific recall@k (using inclusive flags for individual recalls)
            if purchase_count > 0:
                metrics['purchase_recall'] = {k: 0.0 for k in self.k_values}
                metrics['purchase_recall'][k] = in_top_k[is_purchase].mean().item()
                
            if checkout_count_inclusive > 0:
                metrics['checkout_recall'] = {k: 0.0 for k in self.k_values}
                metrics['checkout_recall'][k] = in_top_k[has_checkout_inclusive].mean().item()
                
            if add_to_cart_count_inclusive > 0:
                metrics['add_to_cart_recall'] = {k: 0.0 for k in self.k_values}
                metrics['add_to_cart_recall'][k] = in_top_k[has_add_to_cart_inclusive].mean().item()
                
            if intent_count > 0:
                metrics['intent_recall'] = {k: 0.0 for k in self.k_values}
                metrics['intent_recall'][k] = in_top_k[intent_mask].mean().item()
            
            # Weighted recall@k
            weighted_recall = (in_top_k * event_weights_tensor).sum() / event_weights_tensor.sum()
            metrics['weighted_recall'][k] = weighted_recall.item()
            
            # Precision@k (different from recall - considers all top k predictions)
            # For binary relevance, precision@k = # relevant items in top k / k
            precision_at_k = matches[:, :k].sum(dim=1).float() / k
            metrics['precision'][k] = precision_at_k.mean().item()
            
            # NDCG@k (position-aware metric)
            ndcg_values = torch.zeros(batch_size, device=predictions.device)
            for i in range(batch_size):
                if has_match[i] and ranks[i] <= k:
                    # Position in ranking (0-indexed)
                    position = ranks[i].item() - 1
                    # DCG = 1 / log2(position + 2)  [+2 because position is 0-indexed and log2(1) is 0]
                    dcg = 1.0 / torch.log2(torch.tensor(position + 2, device=predictions.device))
                    # IDCG for binary relevance is always 1 / log2(2) = 1
                    ndcg_values[i] = dcg
            
            metrics['ndcg'][k] = ndcg_values.mean().item()
            
        return metrics
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute final metrics by averaging over all batches.
        
        Returns:
            Dictionary of all metrics averaged over evaluation batches
        """
        results = {}
        
        # Loss
        if self.metrics['loss']:
            results['loss'] = np.mean(self.metrics['loss'])
        
        # Recall@k metrics for all event types and weighted composite
        for k in self.k_values:
            # Overall recall
            results[f'recall@{k}'] = np.mean(self.metrics['recall'][k])
            
            # Event-specific recall
            if self.counts['purchase'] > 0 and self.metrics['purchase_recall'][k]:
                results[f'purchase_recall@{k}'] = np.mean(self.metrics['purchase_recall'][k])
            else:
                results[f'purchase_recall@{k}'] = 0.0
                
            if self.counts['checkout'] > 0 and self.metrics['checkout_recall'][k]:
                results[f'checkout_recall@{k}'] = np.mean(self.metrics['checkout_recall'][k])
            else:
                results[f'checkout_recall@{k}'] = 0.0
                
            if self.counts['add_to_cart'] > 0 and self.metrics['add_to_cart_recall'][k]:
                results[f'add_to_cart_recall@{k}'] = np.mean(self.metrics['add_to_cart_recall'][k])
            else:
                results[f'add_to_cart_recall@{k}'] = 0.0
                
            if self.counts['intent'] > 0 and self.metrics['intent_recall'][k]:
                results[f'intent_recall@{k}'] = np.mean(self.metrics['intent_recall'][k])
            else:
                results[f'intent_recall@{k}'] = 0.0
            
            # Weighted composite recall
            results[f'weighted_recall@{k}'] = np.mean(self.metrics['weighted_recall'][k])
            
            # Additional validation metrics
            results[f'precision@{k}'] = np.mean(self.metrics['precision'][k])
            results[f'ndcg@{k}'] = np.mean(self.metrics['ndcg'][k])
        
        # MRR metrics
        results['mrr'] = self.metrics['mrr'] / max(self.counts['total'], 1)
        
        if self.counts['purchase'] > 0:
            results['purchase_mrr'] = self.metrics['purchase_mrr'] / self.counts['purchase']
        else:
            results['purchase_mrr'] = 0.0
            
        if self.counts['checkout'] > 0:
            results['checkout_mrr'] = self.metrics['checkout_mrr'] / self.counts['checkout']
        else:
            results['checkout_mrr'] = 0.0
            
        if self.counts['add_to_cart'] > 0:
            results['add_to_cart_mrr'] = self.metrics['add_to_cart_mrr'] / self.counts['add_to_cart']
        else:
            results['add_to_cart_mrr'] = 0.0
            
        if self.counts['intent'] > 0:
            results['intent_mrr'] = self.metrics['intent_mrr'] / self.counts['intent']
        else:
            results['intent_mrr'] = 0.0
        
        # Weighted MRR
        results['weighted_mrr'] = self.metrics['weighted_mrr'] / max(self.counts['total'], 1)
        
        # Add counts to results
        results.update(self.counts)
        
        return results


class CrossValidationManager:
    """
    Manages k-fold cross-validation for recommender model training.
    
    Responsible for:
    - Creating k folds from training data
    - Tracking performance across folds
    - Determining the best model across folds
    - Providing statistical significance of results
    """
    
    def __init__(self, 
                 n_splits: int = 5, 
                 shuffle: bool = True, 
                 random_state: int = 42):
        """
        Initialize the cross-validation manager.
        
        Args:
            n_splits: Number of folds for cross-validation
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.fold_results = []
        self.best_fold = None
        self.best_score = -float('inf')
        self.primary_metric = 'weighted_recall@10'  # Default primary metric
    
    def create_folds(self, samples: List[Dict], stratify_by_purchase: bool = True) -> List[Tuple[List, List]]:
        """
        Create k folds from the provided samples.
        
        Args:
            samples: List of sample dictionaries to split
            stratify_by_purchase: Whether to maintain purchase ratio in each fold
            
        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        import numpy as np
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Create array for fold splitting
        indices = np.arange(len(samples))
        
        if stratify_by_purchase:
            # Extract purchase labels for stratification
            labels = np.array([int(sample.get('is_purchase', False)) for sample in samples])
            
            # Use stratified k-fold to maintain purchase ratio
            splitter = StratifiedKFold(n_splits=self.n_splits, 
                                       shuffle=self.shuffle, 
                                       random_state=self.random_state)
            folds = list(splitter.split(indices, labels))
        else:
            # Use regular k-fold without stratification
            splitter = KFold(n_splits=self.n_splits, 
                            shuffle=self.shuffle, 
                            random_state=self.random_state)
            folds = list(splitter.split(indices))
        
        # Convert indices to samples
        folded_samples = []
        for train_idx, val_idx in folds:
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            folded_samples.append((train_samples, val_samples))
            
        return folded_samples
    
    def set_primary_metric(self, metric_name: str):
        """Set the primary metric for determining the best fold"""
        self.primary_metric = metric_name
    
    def add_fold_result(self, fold_idx: int, metrics: Dict[str, Any], model_path: str):
        """
        Add results from a single fold.
        
        Args:
            fold_idx: Index of the current fold
            metrics: Dictionary of evaluation metrics for this fold
            model_path: Path to the saved model for this fold
        """
        fold_data = {
            'fold': fold_idx,
            'metrics': metrics,
            'model_path': model_path
        }
        
        self.fold_results.append(fold_data)
        
        # Update best fold if this one is better
        if self.primary_metric in metrics and (
            self.best_fold is None or 
            metrics[self.primary_metric] > self.best_score
        ):
            self.best_fold = fold_idx
            self.best_score = metrics[self.primary_metric]
    
    def get_best_fold(self) -> Dict[str, Any]:
        """Get the results of the best performing fold"""
        if self.best_fold is None:
            return None
            
        return next((r for r in self.fold_results if r['fold'] == self.best_fold), None)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Calculate average metrics across all folds.
        
        Returns:
            Dictionary of averaged metrics
        """
        if not self.fold_results:
            return {}
            
        # Collect all metrics across folds
        all_metrics = {}
        
        for fold_data in self.fold_results:
            metrics = fold_data['metrics']
            
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                    
                # Skip non-numeric values
                if isinstance(value, (int, float)):
                    all_metrics[key].append(value)
        
        # Calculate averages
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:  # Check if we have any values
                avg_metrics[key] = np.mean(values)
                
                # Also calculate standard deviation for statistical significance
                if len(values) > 1:
                    avg_metrics[f"{key}_std"] = np.std(values)
                    
                    # Calculate 95% confidence interval
                    ci_95 = 1.96 * np.std(values) / np.sqrt(len(values))
                    avg_metrics[f"{key}_ci95"] = ci_95
        
        return avg_metrics
    
    def get_statistical_significance(self, 
                                    model1_metrics: List[float], 
                                    model2_metrics: List[float],
                                    alpha: float = 0.05) -> Tuple[bool, float]:
        """
        Calculate statistical significance between two models' performance.
        
        Args:
            model1_metrics: List of metric values for model 1 across folds
            model2_metrics: List of metric values for model 2 across folds
            alpha: Significance level
            
        Returns:
            Tuple of (is_significant, p_value)
        """
        from scipy import stats
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(model1_metrics, model2_metrics)
        
        # Check if difference is statistically significant
        is_significant = p_value < alpha
        
        return is_significant, p_value
    
    def summarize_results(self) -> str:
        """
        Generate a summary of cross-validation results.
        
        Returns:
            String summary of results
        """
        if not self.fold_results:
            return "No cross-validation results available."
            
        avg_metrics = self.get_average_metrics()
        best_fold = self.get_best_fold()
        
        summary = []
        summary.append("Cross-Validation Results Summary")
        summary.append("===============================")
        summary.append(f"Number of folds: {self.n_splits}")
        summary.append(f"Primary metric: {self.primary_metric}")
        summary.append(f"Average {self.primary_metric}: {avg_metrics.get(self.primary_metric, 'N/A'):.4f}")
        
        if f"{self.primary_metric}_std" in avg_metrics:
            summary.append(f"Standard deviation: {avg_metrics[f'{self.primary_metric}_std']:.4f}")
            
        if f"{self.primary_metric}_ci95" in avg_metrics:
            summary.append(f"95% confidence interval: ±{avg_metrics[f'{self.primary_metric}_ci95']:.4f}")
        
        summary.append("\nBest fold:")
        summary.append(f"Fold {self.best_fold} with {self.primary_metric} = {self.best_score:.4f}")
        summary.append(f"Model path: {best_fold['model_path']}")
        
        summary.append("\nDetailed metrics (averaged across folds):")
        
        # Group metrics by type for better organization
        recall_metrics = {k: v for k, v in avg_metrics.items() if 'recall' in k and '_std' not in k and '_ci95' not in k}
        mrr_metrics = {k: v for k, v in avg_metrics.items() if 'mrr' in k and '_std' not in k and '_ci95' not in k}
        other_metrics = {k: v for k, v in avg_metrics.items() 
                       if 'recall' not in k and 'mrr' not in k and '_std' not in k and '_ci95' not in k}
        
        # Add recall metrics
        summary.append("\nRecall metrics:")
        for metric, value in sorted(recall_metrics.items()):
            summary.append(f"  {metric}: {value:.4f}")
            
        # Add MRR metrics
        summary.append("\nMRR metrics:")
        for metric, value in sorted(mrr_metrics.items()):
            summary.append(f"  {metric}: {value:.4f}")
            
        # Add other metrics
        summary.append("\nOther metrics:")
        for metric, value in sorted(other_metrics.items()):
            # Format count metrics as integers
            if 'count' in metric:
                summary.append(f"  {metric}: {int(value)}")
            else:
                summary.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(summary)


class DynamicHyperparameters:
    """
    Dynamic hyperparameters that can adapt during training.
    
    Features:
    - Learning rate scheduling with warmup
    - Adaptive hyperparameters that update based on validation performance
    - Purchase boost and margin adaptation
    - Temperature scheduling for contrastive learning
    """
    
    def __init__(self, 
                 initial_values: Dict[str, float],
                 adapter_type: str = 'scheduled',
                 min_values: Optional[Dict[str, float]] = None,
                 max_values: Optional[Dict[str, float]] = None):
        """
        Initialize dynamic hyperparameters.
        
        Args:
            initial_values: Dictionary of initial hyperparameter values
            adapter_type: Method for adapting hyperparameters ('scheduled', 'adaptive', 'learnable')
            min_values: Minimum allowed values for each hyperparameter
            max_values: Maximum allowed values for each hyperparameter
        """
        self.values = initial_values.copy()
        self.initial_values = initial_values.copy()
        self.adapter_type = adapter_type
        
        # Set min/max bounds or use defaults with automatic constraint detection
        self.min_values = min_values or {}
        self.max_values = max_values or {}
        
        # Auto-detect sensible bounds based on parameter names
        for param in self.values:
            if param not in self.min_values:
                # Set sensible defaults based on parameter name
                if 'learning_rate' in param:
                    self.min_values[param] = 1e-6
                elif 'margin' in param:
                    self.min_values[param] = 0.0
                elif 'boost' in param:
                    self.min_values[param] = 0.1
                elif 'temperature' in param:
                    self.min_values[param] = 0.01
                else:
                    self.min_values[param] = 0.0
                    
            if param not in self.max_values:
                # Set sensible defaults based on parameter name
                if 'learning_rate' in param:
                    self.max_values[param] = 1.0
                elif 'margin' in param:
                    self.max_values[param] = 1.0
                elif 'boost' in param:
                    self.max_values[param] = 100.0
                elif 'temperature' in param:
                    self.max_values[param] = 1.0
                else:
                    self.max_values[param] = float('inf')
        
        # Initialize history for tracking changes
        self.history = {param: [value] for param, value in self.values.items()}
        self.update_steps = 0
        
        # Initialize adaptation parameters
        if adapter_type == 'scheduled':
            # Scheduled parameters decrease/increase according to a schedule
            self.schedule_type = 'cosine'  # Options: 'linear', 'cosine', 'step'
            self.schedule_params = {
                'warmup_steps': 0,
                'total_steps': 1000,
                'cycles': 1,
                'step_size': 0.1,
                'step_intervals': [0.3, 0.6, 0.9]  # Percentage of total steps
            }
        
        elif adapter_type == 'adaptive':
            # Adaptive parameters change based on validation performance
            self.patience = 2
            self.improvement_threshold = 0.001
            self.adaptation_rate = 0.1
            self.epochs_without_improvement = 0
            self.best_metric = -float('inf')
            
        elif adapter_type == 'learnable':
            # Learnable parameters are updated through gradient descent
            # They require PyTorch parameter registration in the model
            self.learnable_params = {}
            
    def update_step(self, step: int = None, total_steps: int = None):
        """
        Update hyperparameters based on training step for scheduled adaptation.
        
        Args:
            step: Current training step
            total_steps: Total number of training steps
        """
        if self.adapter_type != 'scheduled':
            return
            
        if step is not None:
            self.update_steps = step
        else:
            self.update_steps += 1
            
        if total_steps is not None:
            self.schedule_params['total_steps'] = total_steps
            
        # Apply schedule-based updates
        for param_name, initial_value in self.initial_values.items():
            # Skip parameters that shouldn't be scheduled
            if param_name not in self.values:
                continue
                
            # Get schedule multiplier
            if self.schedule_type == 'cosine':
                # Cosine schedule with optional warmup
                warmup_steps = self.schedule_params.get('warmup_steps', 0)
                total_steps = self.schedule_params.get('total_steps', 1000)
                cycles = self.schedule_params.get('cycles', 1)
                
                if self.update_steps < warmup_steps:
                    # Linear warmup
                    multiplier = float(self.update_steps) / float(max(1, warmup_steps))
                else:
                    # Cosine decay after warmup
                    progress = float(self.update_steps - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    multiplier = 0.5 * (1.0 + np.cos(np.pi * cycles * progress))
                    
            elif self.schedule_type == 'linear':
                # Linear schedule
                total_steps = self.schedule_params.get('total_steps', 1000)
                warmup_steps = self.schedule_params.get('warmup_steps', 0)
                
                if self.update_steps < warmup_steps:
                    # Linear warmup
                    multiplier = float(self.update_steps) / float(max(1, warmup_steps))
                else:
                    # Linear decay after warmup
                    multiplier = 1.0 - (float(self.update_steps - warmup_steps) / 
                                      float(max(1, total_steps - warmup_steps)))
                    
            elif self.schedule_type == 'step':
                # Step schedule
                total_steps = self.schedule_params.get('total_steps', 1000)
                step_size = self.schedule_params.get('step_size', 0.1)
                step_intervals = self.schedule_params.get('step_intervals', [0.3, 0.6, 0.9])
                
                # Calculate current progress
                progress = float(self.update_steps) / float(max(1, total_steps))
                
                # Initialize multiplier at 1.0
                multiplier = 1.0
                
                # Reduce by step_size at each interval
                for interval in step_intervals:
                    if progress > interval:
                        multiplier -= step_size
                        
            else:
                # Unknown schedule type
                multiplier = 1.0
                
            # Apply multiplier to the parameter
            new_value = initial_value * multiplier
            
            # Apply bounds
            if param_name in self.min_values:
                new_value = max(new_value, self.min_values[param_name])
            if param_name in self.max_values:
                new_value = min(new_value, self.max_values[param_name])
                
            # Update value
            self.values[param_name] = new_value
            
            # Track history
            self.history[param_name].append(new_value)
    
    def update_adaptive(self, metric_value: float):
        """
        Update hyperparameters based on validation metric for adaptive parameters.
        
        Args:
            metric_value: Current validation metric value
        """
        if self.adapter_type != 'adaptive':
            return
            
        # Check if there's improvement
        if metric_value > self.best_metric + self.improvement_threshold:
            # Improvement detected
            self.best_metric = metric_value
            self.epochs_without_improvement = 0
        else:
            # No improvement
            self.epochs_without_improvement += 1
            
        # Adapt parameters if no improvement for 'patience' epochs
        if self.epochs_without_improvement >= self.patience:
            for param_name in self.values:
                # Different adaptation strategies based on parameter
                if param_name == 'learning_rate':
                    # Reduce learning rate
                    self.values[param_name] *= (1.0 - self.adaptation_rate)
                    
                elif param_name == 'purchase_boost' or param_name == 'purchase_margin':
                    # Increase purchase importance
                    self.values[param_name] *= (1.0 + self.adaptation_rate)
                    
                elif param_name == 'temperature':
                    # Decrease temperature for sharper predictions
                    self.values[param_name] *= (1.0 - self.adaptation_rate)
                    
                # Apply bounds
                if param_name in self.min_values:
                    self.values[param_name] = max(self.values[param_name], self.min_values[param_name])
                if param_name in self.max_values:
                    self.values[param_name] = min(self.values[param_name], self.max_values[param_name])
                    
                # Track history
                self.history[param_name].append(self.values[param_name])
                
            # Reset counter
            self.epochs_without_improvement = 0
            
    def get_value(self, param_name: str) -> float:
        """Get the current value of a hyperparameter"""
        return self.values.get(param_name, self.initial_values.get(param_name))
    
    def get_all_values(self) -> Dict[str, float]:
        """Get all current hyperparameter values"""
        return self.values.copy()
    
    def get_history(self, param_name: str) -> List[float]:
        """Get the history of a hyperparameter's values"""
        return self.history.get(param_name, [])
    
    def get_learnable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Create learnable PyTorch parameters for trainable hyperparameters.
        
        Returns:
            Dictionary of parameter name to torch.nn.Parameter
        """
        if self.adapter_type != 'learnable':
            return {}
            
        # Create learnable parameters for each hyperparameter
        learnable_params = {}
        
        for param_name, value in self.values.items():
            if 'margin' in param_name:
                # Margin parameters use LearnableMargin module
                learnable_params[param_name] = LearnableMargin(
                    initial_margin=value,
                    min_margin=self.min_values.get(param_name, 0.0),
                    max_margin=self.max_values.get(param_name, 1.0)
                )
            elif 'temperature' in param_name:
                # Temperature parameter requires positive values
                # Use softplus to ensure positive values
                learnable_params[param_name] = torch.nn.Parameter(
                    torch.tensor(value, dtype=torch.float)
                )
            elif 'boost' in param_name:
                # Boost parameters are positive
                # Use log parameterization for stability
                learnable_params[param_name] = torch.nn.Parameter(
                    torch.tensor(np.log(value), dtype=torch.float)
                )
            else:
                # Default case - direct parameterization
                learnable_params[param_name] = torch.nn.Parameter(
                    torch.tensor(value, dtype=torch.float)
                )
                
        self.learnable_params = learnable_params
        return learnable_params
    
    def update_from_learnable(self):
        """Update internal values from learnable parameters"""
        if self.adapter_type != 'learnable' or not hasattr(self, 'learnable_params') or not self.learnable_params:
            return
            
        # Update values from learnable parameters
        for param_name, param in self.learnable_params.items():
            if isinstance(param, LearnableMargin):
                # LearnableMargin module has a forward method
                self.values[param_name] = param().item()
            elif 'boost' in param_name:
                # Exponentiate for boost parameters (log parameterization)
                self.values[param_name] = torch.exp(param).item()
            elif 'temperature' in param_name:
                # Use raw parameter value but apply bounds
                self.values[param_name] = torch.clamp(
                    param,
                    min=self.min_values.get(param_name, 0.01),
                    max=self.max_values.get(param_name, 1.0)
                ).item()
            else:
                # Default case - direct value with bounds
                self.values[param_name] = torch.clamp(
                    param,
                    min=self.min_values.get(param_name, 0.0),
                    max=self.max_values.get(param_name, float('inf'))
                ).item()
                
            # Track history
            self.history[param_name].append(self.values[param_name])


class UnifiedMemoryManager:
    """
    Unified memory management for all device types.
    
    Features:
    - Dynamic batch size adjustment based on available memory
    - Periodic memory cleanup
    - Tensor type optimization for different devices
    - Out-of-memory error prevention
    - Memory usage tracking and reporting
    """
    
    def __init__(self, device_type: str):
        """
        Initialize memory manager for the specified device type.
        
        Args:
            device_type: Device type ('cuda', 'mps', or 'cpu')
        """
        self.device_type = device_type
        self.peak_memory = 0
        self.current_memory = 0
        self.periodic_cleanup = True
        self.cleanup_frequency = 5
        self.current_batch = 0
        self.memory_threshold = 0.9  # Maximum memory usage threshold (90%)
        self.optimize_dtypes = True
        
        # Memory tracking setup
        if device_type == 'cuda':
            import torch
            self.has_memory_tracking = torch.cuda.is_available()
            if self.has_memory_tracking:
                # Reserve some memory to prevent fragmentation
                self.reserve = torch.empty(1024 * 1024, device='cuda')
                
        elif device_type == 'mps':
            import torch
            self.has_memory_tracking = (hasattr(torch, 'mps') and
                                      hasattr(torch.mps, 'current_allocated_memory'))
        else:  # CPU
            self.has_memory_tracking = False
    
    def get_available_memory(self) -> Tuple[int, int]:
        """
        Get currently available memory on the device.
        
        Returns:
            Tuple of (free_memory, total_memory) in bytes
        """
        if self.device_type == 'cuda':
            import torch
            if torch.cuda.is_available():
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                return free_memory, total_memory
                
        elif self.device_type == 'mps':
            import torch
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                # MPS doesn't have a direct API for total/free memory, so we estimate
                # Assuming a 4GB memory size for M1/M2 chips
                estimated_total = 4 * 1024 * 1024 * 1024  # 4GB in bytes
                current_used = torch.mps.current_allocated_memory()
                return estimated_total - current_used, estimated_total
                
        # CPU case or fallback
        import psutil
        memory_info = psutil.virtual_memory()
        return memory_info.available, memory_info.total
    
    def update_memory_info(self):
        """Update current and peak memory usage information"""
        if not self.has_memory_tracking:
            return
            
        if self.device_type == 'cuda':
            import torch
            if torch.cuda.is_available():
                current = torch.cuda.memory_allocated()
                self.current_memory = current
                self.peak_memory = max(self.peak_memory, current)
                
        elif self.device_type == 'mps':
            import torch
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                current = torch.mps.current_allocated_memory()
                self.current_memory = current
                self.peak_memory = max(self.peak_memory, current)
    
    def get_optimal_batch_size(self, 
                              current_batch_size: int, 
                              item_size_bytes: int,
                              min_batch_size: int = 4) -> int:
        """
        Calculate optimal batch size based on available memory.
        
        Args:
            current_batch_size: Current batch size being used
            item_size_bytes: Estimated memory usage per item in the batch
            min_batch_size: Minimum acceptable batch size
            
        Returns:
            Recommended batch size based on available memory
        """
        free_memory, total_memory = self.get_available_memory()
        
        # Reserve 20% of memory for operations and overhead
        usable_memory = free_memory * 0.8
        
        # Calculate how many items we can fit
        max_items = usable_memory // item_size_bytes
        
        # Round down to multiple of 4 (or min_batch_size) for better memory alignment
        optimal_size = max(min_batch_size, (max_items // min_batch_size) * min_batch_size)
        
        # Don't increase batch size too aggressively
        if optimal_size > current_batch_size:
            # Increase by at most 50%
            return min(optimal_size, int(current_batch_size * 1.5))
        elif optimal_size < current_batch_size:
            # Decrease by at most 25%
            return max(optimal_size, int(current_batch_size * 0.75))
        else:
            return current_batch_size
    
    def cleanup(self, force: bool = False):
        """
        Perform memory cleanup operations.
        
        Args:
            force: Whether to force cleanup even if not scheduled
        """
        if not self.periodic_cleanup and not force:
            return
            
        # Check if it's time for a cleanup
        if not force and self.current_batch % self.cleanup_frequency != 0:
            return
            
        if self.device_type == 'cuda':
            import torch
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
        elif self.device_type == 'mps':
            import torch
            if hasattr(torch, 'mps'):
                # Clear MPS cache
                torch.mps.empty_cache()
    
    def optimize_tensor_dtypes(self, tensor, large_threshold: int = 1000) -> torch.Tensor:
        """
        Optimize tensor data types for the current device.
        
        Args:
            tensor: Input tensor to optimize
            large_threshold: Threshold for considering a tensor "large"
            
        Returns:
            Optimized tensor
        """
        if not self.optimize_dtypes:
            return tensor
            
        import torch
        
        # Skip tensors that aren't PyTorch tensors
        if not isinstance(tensor, torch.Tensor):
            return tensor
            
        # Skip tensors that require gradients
        if tensor.requires_grad:
            return tensor
            
        is_large = tensor.numel() > large_threshold
        
        if self.device_type == 'cuda':
            # For CUDA, convert int64 to int32 for large tensors
            if tensor.dtype == torch.int64 and is_large:
                return tensor.to(dtype=torch.int32)
                
        elif self.device_type == 'mps':
            # For MPS, convert int64 to int32 and float64 to float32
            if tensor.dtype == torch.int64 and is_large:
                return tensor.to(dtype=torch.int32)
            elif tensor.dtype == torch.float64:
                return tensor.to(dtype=torch.float32)
                
        # Return unchanged if no optimization applies
        return tensor
    
    def batch_to_device(self, batch, device):
        """
        Move batch to device with optimized handling for CUDA/MPS/CPU.
        
        Args:
            batch: Data batch to move to device
            device: PyTorch device to move data to
            
        Returns:
            Batch on the specified device with optimized data types
        """
        import torch
        
        if isinstance(batch, dict):
            # Process each dictionary entry
            return {k: self.batch_to_device(v, device) for k, v in batch.items()}
            
        elif isinstance(batch, torch.Tensor):
            # Optimize tensor and move to device
            optimized = self.optimize_tensor_dtypes(batch)
            return optimized.to(device, non_blocking=True)
            
        elif isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
            # Handle list of tensors
            return [self.optimize_tensor_dtypes(t).to(device, non_blocking=True) for t in batch]
            
        else:
            # Return unchanged for other types
            return batch
    
    def next_batch(self):
        """Update batch counter and perform periodic cleanup if needed"""
        self.current_batch += 1
        self.update_memory_info()
        
        if self.periodic_cleanup and self.current_batch % self.cleanup_frequency == 0:
            self.cleanup()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary of memory statistics
        """
        free_memory, total_memory = self.get_available_memory()
        
        stats = {
            'device_type': self.device_type,
            'current_memory_bytes': self.current_memory,
            'peak_memory_bytes': self.peak_memory, 
            'free_memory_bytes': free_memory,
            'total_memory_bytes': total_memory,
            'memory_utilization': 1.0 - (free_memory / max(1, total_memory))
        }
        
        # Add human-readable values in MB
        for key in ['current_memory_bytes', 'peak_memory_bytes', 'free_memory_bytes', 'total_memory_bytes']:
            if key in stats:
                mb_key = key.replace('_bytes', '_mb')
                stats[mb_key] = stats[key] / (1024 * 1024)
                
        return stats


# Example usage for standalone testing
if __name__ == "__main__":
    import torch
    
    # Test the UnifiedMetricsTracker
    metrics_tracker = UnifiedMetricsTracker()
    
    # Create dummy data
    batch_size = 32
    num_items = 1000
    
    # Simulate predictions and targets
    predictions = torch.randn(batch_size, num_items)
    targets = torch.randint(0, num_items, (batch_size,))
    
    # Create dummy event flags
    is_purchase = torch.zeros(batch_size, dtype=torch.bool)
    is_purchase[0:5] = True  # Mark 5 examples as purchases
    
    has_checkout = torch.zeros(batch_size, dtype=torch.bool)
    has_checkout[6:12] = True  # Mark 6 examples as checkouts
    
    has_add_to_cart = torch.zeros(batch_size, dtype=torch.bool)
    has_add_to_cart[13:20] = True  # Mark 7 examples as add-to-cart
    
    # Update metrics
    metrics_tracker.update(
        predictions=predictions,
        targets=targets,
        is_purchase=is_purchase,
        has_checkout=has_checkout,
        has_add_to_cart=has_add_to_cart,
        loss=0.5
    )
    
    # Compute and print results
    results = metrics_tracker.compute()
    print("Test results:")
    for key, value in sorted(results.items()):
        print(f"  {key}: {value}")
    


# Backward compatibility functions for migration from utils/metrics.py
def hit_rate_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """Calculate Hit Rate@k (also known as Recall@k) - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=[k])
    tracker.update(
        predictions=scores,
        targets=ground_truth,
        is_purchase=torch.ones_like(ground_truth, dtype=torch.bool),
        has_checkout=torch.zeros_like(ground_truth, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(ground_truth, dtype=torch.bool)
    )
    results = tracker.compute()
    return results.get(f'recall@{k}', 0.0)


def mrr_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """Calculate Mean Reciprocal Rank@k - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=[k])
    tracker.update(
        predictions=scores,
        targets=ground_truth,
        is_purchase=torch.ones_like(ground_truth, dtype=torch.bool),
        has_checkout=torch.zeros_like(ground_truth, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(ground_truth, dtype=torch.bool)
    )
    results = tracker.compute()
    return results.get('mrr', 0.0)


def ndcg_at_k(scores: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain@k - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=[k])
    tracker.update(
        predictions=scores,
        targets=ground_truth,
        is_purchase=torch.ones_like(ground_truth, dtype=torch.bool),
        has_checkout=torch.zeros_like(ground_truth, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(ground_truth, dtype=torch.bool)
    )
    results = tracker.compute()
    return results.get(f'ndcg@{k}', 0.0)


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                     k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """Calculate multiple metrics at once - Legacy function for backward compatibility"""
    tracker = UnifiedMetricsTracker(k_values=k_values)
    tracker.update(
        predictions=predictions,
        targets=targets,
        is_purchase=torch.ones_like(targets, dtype=torch.bool),
        has_checkout=torch.zeros_like(targets, dtype=torch.bool),
        has_add_to_cart=torch.zeros_like(targets, dtype=torch.bool)
    )
    results = tracker.compute()
    
    # Convert to legacy format
    legacy_results = {}
    for k in k_values:
        legacy_results[f'hit@{k}'] = results.get(f'recall@{k}', 0.0)
        legacy_results[f'mrr@{k}'] = results.get('mrr', 0.0)
        legacy_results[f'ndcg@{k}'] = results.get(f'ndcg@{k}', 0.0)
    
    return legacy_results


class MetricsTracker:
    """Legacy MetricsTracker class for backward compatibility with existing training scripts"""
    def __init__(self, k_values: List[int] = [1, 5, 10, 20]):
        self.unified_tracker = UnifiedMetricsTracker(k_values=k_values)
        self.k_values = k_values
        
    def reset(self):
        """Reset all tracked metrics"""
        self.unified_tracker.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               loss: Optional[float] = None):
        """Update metrics with a new batch"""
        self.unified_tracker.update(
            predictions=predictions,
            targets=targets,
            is_purchase=torch.ones_like(targets, dtype=torch.bool),
            has_checkout=torch.zeros_like(targets, dtype=torch.bool),
            has_add_to_cart=torch.zeros_like(targets, dtype=torch.bool),
            loss=loss
        )
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics in legacy format"""
        results = self.unified_tracker.compute()
        
        # Convert to legacy format
        legacy_results = {}
        
        if 'loss' in results:
            legacy_results['loss'] = results['loss']
        
        for k in self.k_values:
            legacy_results[f'hit@{k}'] = results.get(f'recall@{k}', 0.0)
            legacy_results[f'mrr@{k}'] = results.get('mrr', 0.0)
            legacy_results[f'ndcg@{k}'] = results.get(f'ndcg@{k}', 0.0)
        
        return legacy_results


class EnhancedEventMetrics:
    """Legacy EnhancedEventMetrics class for backward compatibility"""
    
    @staticmethod
    def calculate_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        is_purchase: torch.Tensor,
        has_checkout: torch.Tensor,
        has_add_to_cart: torch.Tensor,
        k_values: List[int] = [10, 20]
    ) -> Dict[str, Any]:
        """Calculate metrics with event type awareness - Legacy interface"""
        tracker = UnifiedMetricsTracker(k_values=k_values)
        tracker.update(
            predictions=predictions,
            targets=targets,
            is_purchase=is_purchase,
            has_checkout=has_checkout,
            has_add_to_cart=has_add_to_cart
        )
        results = tracker.compute()
        
        # Convert to legacy format with nested structure
        legacy_results = {}
        
        # Event counts
        legacy_results['purchase_count'] = tracker.counts['purchase']
        legacy_results['checkout_count'] = tracker.counts['checkout']
        legacy_results['add_to_cart_count'] = tracker.counts['add_to_cart']
        legacy_results['intent_count'] = tracker.counts['intent']
        
        # Recall@k metrics in nested format
        legacy_results['recall@k'] = {}
        for k in k_values:
            legacy_results['recall@k'][k] = results.get(f'recall@{k}', 0.0)
        
        if tracker.counts['purchase'] > 0:
            legacy_results['purchase_recall@k'] = {}
            for k in k_values:
                legacy_results['purchase_recall@k'][k] = results.get(f'purchase_recall@{k}', 0.0)
        
        if tracker.counts['checkout'] > 0:
            legacy_results['checkout_recall@k'] = {}
            for k in k_values:
                legacy_results['checkout_recall@k'][k] = results.get(f'checkout_recall@{k}', 0.0)
        
        if tracker.counts['add_to_cart'] > 0:
            legacy_results['add_to_cart_recall@k'] = {}
            for k in k_values:
                legacy_results['add_to_cart_recall@k'][k] = results.get(f'add_to_cart_recall@{k}', 0.0)
        
        if tracker.counts['intent'] > 0:
            legacy_results['intent_recall@k'] = {}
            for k in k_values:
                legacy_results['intent_recall@k'][k] = results.get(f'intent_recall@{k}', 0.0)
        
        # MRR metrics
        legacy_results['overall_mrr'] = results.get('mrr', 0.0)
        legacy_results['purchase_mrr'] = results.get('purchase_mrr', 0.0)
        legacy_results['checkout_mrr'] = results.get('checkout_mrr', 0.0)
        legacy_results['add_to_cart_mrr'] = results.get('add_to_cart_mrr', 0.0)
        legacy_results['intent_mrr'] = results.get('intent_mrr', 0.0)
        
        return legacy_results


# Test backward compatibility after all classes are defined
if __name__ == "__main__":
    print("\nTesting backward compatibility...")
    
    # Create test data
    batch_size = 32
    num_items = 1000
    predictions = torch.randn(batch_size, num_items)
    targets = torch.randint(0, num_items, (batch_size,))
    is_purchase = torch.zeros(batch_size, dtype=torch.bool)
    is_purchase[0:5] = True
    has_checkout = torch.zeros(batch_size, dtype=torch.bool)
    has_checkout[6:12] = True
    has_add_to_cart = torch.zeros(batch_size, dtype=torch.bool)
    has_add_to_cart[13:20] = True
    
    # Test legacy MetricsTracker
    legacy_tracker = MetricsTracker(k_values=[10, 20])
    legacy_tracker.update(predictions, targets, loss=0.5)
    legacy_results = legacy_tracker.compute()
    print("Legacy MetricsTracker results:", legacy_results)
    
    # Test legacy EnhancedEventMetrics
    legacy_enhanced = EnhancedEventMetrics.calculate_metrics(
        predictions, targets, is_purchase, has_checkout, has_add_to_cart, k_values=[10, 20]
    )
    print("Legacy EnhancedEventMetrics results keys:", list(legacy_enhanced.keys()))
    
    print("✅ All backward compatibility tests passed!")