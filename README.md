# Anime-Generator
This project is base on DCGAN, which is also called deep convolutional generative adversarial network
# DCGAN - Deep Convolutional Generative Adversarial Network
  In recnet years, there have been tremendous advancements in the field of machine learning. Lately a amount of research has been dedicated to the usage of generative models in the field of computer vision and iamge classification. The original model proposed by Goodfellow, which is composed of a adversarial nets and discriminative network. However, there are also existing some limitation of original GAN, such as it is unstable to train and resulting in generators that produce nonsensical output.
  
  DCGAN, which is also called deep convolutional generative adversarial networks having certain architectural constraints, is one of the successful network design for GAN. Compared with GAN, this model is more stable to train.
The following points are the core to the DCGAN which adopts some changes to CNN architectures.
- Replace any pooling layers with strided convolution (discriminator) and fractional-strided convolutions(generator), benifitting in learning spatial unsampling
- Remove fully connected hidden layers on top of convolutional features for deeper architectures.
- Use batch normalization which stabilizes learning in both the generator and the discriminator(except the generator output layer and the discriminator input layer)
- Use ReLU activation in generator for all layers except for the output, which uses Tanh function. The bounded activation allows the model to learn more quickly to saturate and cover the color space of the training distribution.

The followiing picture shows the structure of DCGAN generator used for LSUN scene modeling.
![image](Image/图片.png)
