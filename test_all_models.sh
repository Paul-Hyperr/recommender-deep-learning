#!/bin/bash

# Test script to run all four NATR models with small data
# This script runs each model for a minimal number of epochs to verify functionality

echo "========================================================="
echo "Testing NATR models with consolidated training utilities"
echo "========================================================="

# Set common small data parameters for quick testing
# Use fastest mode with minimal epochs and small batch size for quick tests
COMMON_ARGS="--mode fastest --epochs 1 --batch-size 32"
PRETRAIN_ARGS="--mode fastest --batch-size 32 --pretrain-epochs 1 --finetune-epochs 1"

# Create timestamp for test run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create necessary directories
mkdir -p test_outputs

# Set success/failure counters
SUCCESS_COUNT=0
FAILURE_COUNT=0

echo ""
echo "1. Testing Base NATR Model"
echo "========================================================="
python scripts/train_natr.py $COMMON_ARGS 2>&1 | tee "test_outputs/train_natr_test_$TIMESTAMP.txt"
if [ $? -eq 0 ]; then
  echo "✓ Base NATR test completed successfully"
  SUCCESS_COUNT=$((SUCCESS_COUNT+1))
else
  echo "✗ Base NATR test failed"
  FAILURE_COUNT=$((FAILURE_COUNT+1))
fi

echo ""
echo "2. Testing Contrastive NATR Model"
echo "========================================================="
python scripts/train_natr_contrastive.py $COMMON_ARGS 2>&1 | tee "test_outputs/train_natr_contrastive_test_$TIMESTAMP.txt"
if [ $? -eq 0 ]; then
  echo "✓ Contrastive NATR test completed successfully"
  SUCCESS_COUNT=$((SUCCESS_COUNT+1))
else
  echo "✗ Contrastive NATR test failed"
  FAILURE_COUNT=$((FAILURE_COUNT+1))
fi

echo ""
echo "3. Testing Pretrain-Finetune NATR Model"
echo "========================================================="
python scripts/train_natr_pretrain_finetune.py $PRETRAIN_ARGS 2>&1 | tee "test_outputs/train_natr_pretrain_finetune_test_$TIMESTAMP.txt"
if [ $? -eq 0 ]; then
  echo "✓ Pretrain-Finetune NATR test completed successfully"
  SUCCESS_COUNT=$((SUCCESS_COUNT+1))
else
  echo "✗ Pretrain-Finetune NATR test failed"
  FAILURE_COUNT=$((FAILURE_COUNT+1))
fi

echo ""
echo "4. Testing Checkout-Enhanced NATR Model"
echo "========================================================="
python scripts/train_natr_checkout_enhanced.py $COMMON_ARGS 2>&1 | tee "test_outputs/train_natr_checkout_test_$TIMESTAMP.txt"
if [ $? -eq 0 ]; then
  echo "✓ Checkout-Enhanced NATR test completed successfully"
  SUCCESS_COUNT=$((SUCCESS_COUNT+1))
else
  echo "✗ Checkout-Enhanced NATR test failed"
  FAILURE_COUNT=$((FAILURE_COUNT+1))
fi

echo ""
echo "========================================================="
echo "Test Summary: $SUCCESS_COUNT succeeded, $FAILURE_COUNT failed"
echo "Test logs saved to: test_outputs/train_natr_*_test_$TIMESTAMP.txt"
echo "========================================================="

# If all tests succeeded, run model comparison
if [ $SUCCESS_COUNT -gt 1 ] && [ $FAILURE_COUNT -eq 0 ]; then
  echo ""
  echo "5. Running Model Comparison"
  echo "========================================================="
  echo "All training tests completed successfully. Comparing model performance..."
  
  # Run model comparison script
  python scripts/compare_models.py --output-file "test_outputs/model_comparison_$TIMESTAMP.txt" --csv-export "test_outputs/model_comparison_$TIMESTAMP.csv" 2>&1 | tee "test_outputs/comparison_$TIMESTAMP.txt"
  
  if [ $? -eq 0 ]; then
    echo "✓ Model comparison completed successfully"
    echo ""
    echo "Results saved to:"
    echo "  - test_outputs/model_comparison_$TIMESTAMP.txt (detailed report)"
    echo "  - test_outputs/model_comparison_$TIMESTAMP.csv (CSV data)"
    echo "  - test_outputs/comparison_$TIMESTAMP.txt (comparison log)"
  else
    echo "✗ Model comparison failed (models may not have been saved properly)"
  fi
fi

echo ""
echo "========================================================="
echo "All operations completed!"
echo "Check test_outputs/ directory for all results and comparisons"
echo "========================================================="

# Return overall success/failure
if [ $FAILURE_COUNT -gt 0 ]; then
  echo "Some tests failed. See logs for details."
  exit 1
else
  echo "All tests completed successfully!"
  exit 0
fi