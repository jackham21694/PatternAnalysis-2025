## TimeGAN for Synthetic LOBSTER Financial Data Generation

## Background
Generative Adverarial Networks have emerged as one of the more popular machine learning frameworks in recent years, given
the increasing desire for generative AI, and synthetic data producttion. Popular networks such as the StyleGAN and widely reknowned
transformers such as ChatGPT have taken the world by storm, but both have unique weakness when it comes to handling time-series data.
Traditional GAN cannot capture the temporal dyamics of time-series data, and transformers are purely deterministic, not built for syntethic data generation. In 2020 a new model entered the atmosphere known as the TimeGAN, designed for synthetic data generation, focusing on temporal data dynamics, through the addition of supervised losses.

## Algorithm Description and Architecture

TimeGAN is designed around the basic unsupervised GAN setup, with the addition of a supervisor loss as used in
autoregressive models. There are 4 main components: Autoencoder, Supervisor, Generator,
and Discriminator. 

AutoEncoder: Split into a traditional embedder and recovery structure, the implementation of the autoencoder
             can be parameterised by any model as long as it is autoregressive and 'obeys casual
             ordering.' Our autoencoder uses 3 stacked GRU layers, and a fully connected layer for
             both the embedder and recovery components. The autoencoder aims to condense the high-
             dimensional time-series data to a lower-dimensional latent space.

Supervisor: This component is what makes the TimeGAN understand complex time-step relationships
            within time-series data. It is parametires by 2 stacked GRU layers, and a dense
            fully connected layer. Given a sequence of timesteps inside the latent space, 
            it is trained to predict the next step inside the latent space for each timestep, 
            in the sequence. 

Generator: The generator takes random noise in the shape of the latent space, and generates
           a synthetic sample within the latent space. The random noise is generated via the
           Weiner process, which adheres to temporal behaviour. It is also built on
           3 stacked GRU layers with a fully connected layer.

Discriminator: The discriminator also operates in the embedding space, and attempts to
               distinguish between real and fake data sequences. It is also built on
               3 stacked GRU layers with a fully connected layer.

![TimeGAN Architecture Image](recognition\TimeGAN46974248\assets\TimeGANArchitecture.jpg)

## Training Process

The supervisor loss is then used in both the generator and discriminator components to help our GAN with temporal dynamics.

![TimeGAN Training Image](recognition\TimeGAN46974248\assets\TimeGANTraining.jpg)



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
https://www.jpmorgan.com/content/dam/jpm/cib/complex/content/technology/ai-research-publications/pdf-12.pdf
https://www.tensorflow.org/guide/migrate#migrate-from-tensorflow-1x-to-tensorflow-2
https://github.com/Jeonghwan-Cheon/lob-deep-learning
https://arxiv.org/abs/1808.03668?utm_source=chatgpt.com
https://link.springer.com/article/10.1007/s10462-024-10715-4?utm_source=chatgpt.com