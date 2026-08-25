Reproduce DISEF on CIFAR100
1. choose 16 images per class in CIFAR100 (1600 real images)
2. get description for each image using pretrained Qwen VLM 2.5
3. create synthetic image, using Stable Diffusion 2.1, add noise to latent of real image for starting point, using description of other object in same class as guidance. Only allow image get high score from Qwen VLM 2.5 (cosine similarity to its class) (64 images per class = 6400)
4. finetune Qwen VLM 2.5 on task image classification (input image, output image embedding, then calculate cosine similarity to all classes). loss function is weighted CE (use their loss)
