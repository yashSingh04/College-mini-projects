Conditional GAN for Neural Colorization (gray to color image)
Neural colorization is a technique that uses neural networks, particularly convolutional neural networks (CNNs), to add color to grayscale images. Instead of manually adding color, the network learns to predict and generate realistic color distributions based on patterns, textures, and context within the image.

Using adversarial learning, we trained a conditional GAN on the Mini-ImageNet dataset. The generator is a U-Net model that colorizes grayscale images, conditioned on luminance values. It downscales and then upscales the image, with skip connections to retain spatial details, and includes self-attention in the decoder to capture long-range dependencies for more coherent colorization. The discriminator is a binary classifier that distinguishes real color images from generated ones, using downsampling convolutional layers to identify inconsistencies.

To help the network capture contextual information during colorization, random patch removal was introduced during training. This technique encourages the model to infer missing information from surrounding areas, enhancing its ability to generate coherent and contextually accurate colors.

Below is the result of Neural Colorization.![colorization1](https://github.com/user-attachments/assets/9b9bce56-373c-4f66-9b59-cd0de454ba81)
![colorization2](https://github.com/user-attachments/assets/2b313e0a-6da7-49a7-8daa-a1a7f33c915c)



    
