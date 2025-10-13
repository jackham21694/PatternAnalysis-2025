# Generative Time-Series Model for LOB Data

Train a generative time-series model, such as **TimeGAN**, to generate synthetic sequences of **limit order book (LOB)** events using the **LOBSTER dataset** (use AMZN level 10 data).  

## Evaluation Metrics

Evaluate the generated sequences on a held-out test split using the following metrics:

- **Distribution similarity**:  
  KL divergence ≤ 0.1 between the generated and real spread and midprice return distributions.

- **Visual similarity**:  
  SSIM > 0.6 between heatmaps of generated vs real LOB depth snapshots.

## Report Requirements

Include the following details in your report:

- Model architecture and parameter count
- Training strategy:
  - Full model
  - Variants (e.g., adversarial-only or supervised-only losses)
- GPU type and VRAM
- Number of epochs
- Total training time
- 3–5 representative heatmap visualizations comparing generated vs real order books
- Short error analysis paragraph discussing where the synthetic LOBs succeed and fail


