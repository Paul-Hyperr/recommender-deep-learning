# Neural Attentive Travel Recommender (NATR)

This repository contains an implementation of the Neural Attentive Travel Recommender (NATR) model, specifically optimized for travel package recommendations.

## Overview

NATR is a deep learning model for travel package recommendation that uses:
- Multi-view attention for package representation (title, destination, categories)
- Short-term and long-term user preference modeling
- Attention mechanisms at multiple levels
- Natural event learning for better recommendations

## Features

- Multi-view travel package representation (title, location, categories)
- User preference modeling with short-term and long-term interests
- Gated fusion for preference combination
- Support for various title embedding methods (OpenAI, FastText)
- Enhanced performance modes for different hardware

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/travel-recommender.git
cd travel-recommender

# Install dependencies
pip install -r requirements.txt
```

## Performance Modes

The training script supports multiple performance modes:

- `balanced`: Default balanced mode (best for most cases)
- `fastest`: Maximum speed at the cost of some accuracy
- `accurate`: Maximum accuracy at the cost of training speed
- `apple_silicon`: Optimized specifically for Apple M1/M2/M3 chips

## Usage

### Training the Model

```bash
# Train with default settings
python scripts/train_natr.py

# Train with specific performance mode
python scripts/train_natr.py --mode apple_silicon

# Advanced settings
python scripts/train_natr.py --mode fastest --batch-size 64 --epochs 20 --workers 4
```

### Apple Silicon Optimization

This implementation includes specific optimizations for Apple Silicon (M1/M2/M3):

- Memory-efficient tensor operations
- MPS-specific optimizations in data loading and training
- Gradient accumulation to handle larger models
- Periodic memory cleanup to prevent OOM errors
- Modified batch sizes and model dimensions for Apple GPUs

To use these optimizations:

```bash
python scripts/train_natr.py --mode apple_silicon
```

### Making Recommendations

```bash
python scripts/recommendations.py --model_path checkpoints/natr/best_model.pth --user_id 12345
```

## Model Architecture

The NATR model consists of three main components:

1. **Travel Package Encoder**: Represents packages with multi-view attention
   - Title representation using embeddings
   - Geographical representation using coordinates
   - Categorical representation (country, category, theme)
   
2. **User Encoder**: Models user preferences at different time scales
   - Short-term interests from recent interactions
   - Long-term interests from historical interactions
   - User-specific embedding

3. **Gated Fusion Network**: Combines preferences with adaptive weights
   - Learns importance of short vs. long-term preferences
   - Adapts to different user behavior patterns

## License

This project is licensed under the MIT License - see the LICENSE file for details.