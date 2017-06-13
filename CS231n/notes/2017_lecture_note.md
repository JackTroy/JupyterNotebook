# 2017 Lecture Note 
## Lecture Note 11 Detection and Segmentation
### Instance Segmentation
- Mask R-CNN: a further research based on Faster R-CNN, with a new network on the top

## Lecture Note 13 Generative Models(not understood)

### PixelRNN and PixelCNN
- Explicit density model, optimizes exact likelihood, good samples. But inefficient sequential generation.

### Variational Autoencoders
- Optimize variational lower bound on likelihood. Useful latent representation, inference queries. But current sample quality not the best.

### Generative Adversarial Networks
- Game-theoretic approach, best samples! But can be tricky and unstable to train, no inference queries.

## Lecture Note 15 Efficient Methods and Hardware for Deep Learning or Real World Use

### Part 1: Algorithms for Efficient Inference
1. pruning, abondon some connections then retrain to recover accuracy
2. Weight Sharing, share weight, reduce bits
3. Quantization, 
4. Low Rank Approximation
5. Binary / Ternary Net (2 or 3)
6. Winograd Transformation

### Part 2: Hardware for Efficient Inference
- minimize memory access

### Part 3: Efficient Training — Algorithms
1. Parallelization
2. Mixed Precision with FP16 and FP32
3. Model Distillation
4. DSD: Dense-Sparse-Dense Training

### Part 4: Hardware for Efficient Training
- all sorts of powerful hardware