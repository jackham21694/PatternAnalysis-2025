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


## Current Dependencies Required:

- numpy
- tensorflow
- scikit-learn 



## References
https://numpy.org/doc
https://www.tensorflow.org/api_docs/
https://proceedings.neurips.cc/paper_files/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf
https://github.com/jsyoon0823/TimeGAN
https://lobsterdata.com/info/DataSamples.php
https://ydata.ai/resources/synthetic-time-series-data-a-gan-approach
https://notes.yeshiwei.com/_downloads/2ce792aff8596ea9453a9714f39d957a/Machine_Learning_for_Algorithmic_Trading_Predictive.pdf
https://github.com/stefan-jansen/machine-learning-for-trading/tree/main
https://repository.lib.fsu.edu/islandora/object/fsu:770676
https://bechirtr97.medium.com/enhancing-timegan-with-language-model-architectures-autoregressive-transformers-and-positional-bc6a7024e047