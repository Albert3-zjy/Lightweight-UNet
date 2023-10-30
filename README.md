# Lightweight-UNet

这个轻量级UNet的架构实现继承自 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 的UNet实现。

我们对UNet的架构进行轻量化，对网络每一层的通道数等做了适当缩减，具体是缩减为原来的```1/4```。

我们的轻量级UNet相比传统UNet训练速度和预测速度都有明显提升。

文件夹Lightweight-UNet-keras下即为轻量级U-Net的keras版本
