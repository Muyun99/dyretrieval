# dyretrieval
基于 PyTorch 的图像检索框架

主要基于 [Bag of Tricks and A Strong Baseline for Deep Person Re-identification.](https://github.innominds.com/Muyun99/Awesome-Bag-of-Tricks#reId-or-image-retrieval) 尝试相关基础 trick

主要的训练测试脚本在 `train.sh` / `test.sh` 中


将训练所使用的 csv 文件格式整理如下即可运行

```
filename,label
train/000232.jpg,0
train/003552.jpg,0
train/000814.jpg,1
train/013765.jpg,1
train/001429.jpg,2
train/014834.jpg,2
train/012795.jpg,3
train/015860.jpg,3
train/012123.jpg,4

```

可能有效的 trick
- 将特征提取模块改成 swin transformer，可能会有奇效