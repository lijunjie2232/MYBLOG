## Diffusion Models in PyTorch

While Diffusion Models have not yet been democratized to the same degree as other older architectures/approaches in Machine Learning, there are still implementations available for use. The easiest way to use a Diffusion Model in PyTorch is to use the `denoising-diffusion-pytorch` package, which implements an image diffusion model like the one discussed in this article. To install the package, simply type the following command in the terminal:

```javascript
pip install denoising_diffusion_pytorch
```

### Minimal Example

To train a model and generate images, we first import the necessary packages:

```javascript
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
```

Next, we define our network architecture, in this case a U-Net. The `dim` parameter specifies the number of feature maps before the first down-sampling, and the `dim_mults` parameter provides multiplicands for this value and successive down-samplings:

```javascript
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)
```

Now that our network architecture is defined, we need to define the Diffusion Model itself. We pass in the U-Net model that we just defined along with several parameters - the size of images to generate, the number of timesteps in the diffusion process, and a choice between the L1 and L2 norms.

```javascript
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)
```

Now that the Diffusion Model is defined, it's time to train. We generate random data to train on, and then train the Diffusion Model in the usual fashion:

```javascript
training_images = torch.randn(8, 3, 128, 128)
loss = diffusion(training_images)
loss.backward()
```

Once the model is trained, we can finally generate images by using the `sample()` method of the `diffusion` object. Here we generate 4 images, which are only noise given that our training data was random:

```javascript
sampled_images = diffusion.sample(batch_size = 4)
```

![](https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/67bcdd46d02c1aa00827599e_image-21.png)

### Training on Custom Data

The `denoising-diffusion-pytorch` package also allow you to train a diffusion model on a specific dataset. Simply replace the `'path/to/your/images'` string with the dataset directory path in the `Trainer()` object below, and change `image_size` to the appropriate value. After that, simply run the code to train the model, and then sample as before. Note that PyTorch must be compiled with CUDA enabled in order to use the `Trainer` class:

```javascript
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()
```

Below you can see progressive denoising from multivariate Gaussian noise to MNIST digits akin to reverse diffusion:

![](https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/67b971061998063818b0ce91_movie.gif)

## Final Words

Diffusion Models are a conceptually simple and elegant approach to the problem of generating data. Their State-of-the-Art results combined with non-adversarial training has propelled them to great heights, and further improvements can be expected in the coming years given their nascent status. In particular, Diffusion Models have been found to be essential to the performance of cutting-edge models like [**DALL-E 2**](https://www.assemblyai.com/blog/how-dall-e-2-actually-works/).

## References

\[1\] [**Deep Unsupervised Learning using Nonequilibrium Thermodynamics**](https://arxiv.org/abs/1503.03585?ref=assemblyai.com)

\[2\] [**Generative Modeling by Estimating Gradients of the Data Distribution**](https://arxiv.org/abs/1907.05600?ref=assemblyai.com)

\[3\] [**Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2006.11239?ref=assemblyai.com)

\[4\] [**Improved Techniques for Training Score-Based Generative Models**](https://arxiv.org/abs/2006.09011?ref=assemblyai.com)

\[5\] [**Improved Denoising Diffusion Probabilistic Models**](https://arxiv.org/abs/2102.09672?ref=assemblyai.com)

\[6\] [**Diffusion Models Beat GANs on Image Synthesis**](https://arxiv.org/abs/2105.05233?ref=assemblyai.com)

\[7\] [**GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models**](https://arxiv.org/abs/2112.10741?ref=assemblyai.com)

\[8\] [**Hierarchical Text-Conditional Image Generation with CLIP Latents**](https://arxiv.org/abs/2204.06125?ref=assemblyai.com)

Title goes here

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

[Button Text](#)

#### Related posts

![](https://cdn.prod.website-files.com/67a08d9d7d19f8fb63692894/67a0fb99445797bcd265154a_16x9%20cover%20full.webp)![](https://cdn.prod.website-files.com/67a08d9d7d19f8fb63692894/67a0fb99445797bcd265154a_16x9%20cover%20full.webp)

May 12, 2022

### Introduction to Diffusion Models for Machine Learning

[](/blog/diffusion-models-for-machine-learning-introduction)

By 

Ryan O'Connor

, 

Senior Developer Educator

![](https://cdn.prod.website-files.com/67a08d9d7d19f8fb63692894/67a0fb99445797bcd265154a_16x9%20cover%20full.webp)![](https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/69c17de012a828252a36b41d_Blog-AssemblyAI%20Named%20Leader%20in%20G2%27s%20Spring%202026%20Voice%20Grid.png)

March 23, 2026

### AssemblyAI Named a Leader in G2’s Spring 2026 Voice Recognition Report

[](/blog/g2-spring-2026-voice-recognition-report)

By 

Devon Malloy

, 

Staff Growth Manager

No items found.

![](https://cdn.prod.website-files.com/67a08d9d7d19f8fb63692894/67a0fb99445797bcd265154a_16x9%20cover%20full.webp)![](https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/689a04bef02a18be1d7eccb7_Blog-Build%20an%20AI%20Assistant%20for%20Meeting%20Scheduling%20with%20LiveKit%20(1).png)

August 11, 2025

### Build and deploy real-time AI voice agents using LiveKit and AssemblyAI

[](/blog/build-and-deploy-real-time-ai-voice-agents-using-livekit-and-assemblyai)

By 

Kelsey Foster

, 

Growth

[Streaming Speech-to-Text](/topic/streaming-speech-to-text)

[AI voice agents](/topic/ai-agents)

![](https://cdn.prod.website-files.com/67a08d9d7d19f8fb63692894/67a0fb99445797bcd265154a_16x9%20cover%20full.webp)![](https://cdn.prod.website-files.com/67a08d9d7d19f8fb63692894/67a0fb99445797bcd265154a_16x9%20cover%20full.webp)

November 15, 2023

### 7 best practices for product teams to consider when building with AI

[](/blog/7-best-practices-for-building-better-products-with-ai)

By 

Kelsey Foster

, 

Growth

[Product Management](/topic/product-management)

![](https://cdn.prod.website-files.com/67a08d9d7d19f8fb63692894/67a0fb99445797bcd265154a_16x9%20cover%20full.webp)![](https://cdn.prod.website-files.com/67a1e6de2f2eab2e125f8b9a/68f0e397419928ff07ee587a_Blog-Speech%20recognition%20in%20the%20browser%20using%20Web%20Speech%20API.png)

October 16, 2025

### Speech recognition in the browser using Web Speech API

[](/blog/speech-recognition-javascript-web-speech-api)

By 

Patrick Loeber

, 

Senior Developer Advocate

[Tutorial](/topic/tutorial)

[JavaScript](/topic/javascript)

[Automatic Speech Recognition](/topic/automatic-speech-recognition)

## Embedded Content