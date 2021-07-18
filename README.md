# Learned-image-compression-with-GLLMM
The code of the Learned Image Compression with Discretized Gaussian-Laplacian-Logistic Mixture Model and Concatenated Residual Modules.
Our code is based on Cheng2020's paper named  Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules. [arXiv](https://arxiv.org/abs/2001.01568), CVPR2020. 
# Learned Image Compression with Discretized Gaussian-Laplacian-Logistic Mixture Model and Concatenated Residual Modules.
This repository contains the code for reproducing the results with trained models, in the following paper:
Learned Image Compression with Discretized Gaussian-Laplacian-Logistic Mixture Model and Concatenated Residual Modules. [arXiv](https://arxiv.org/pdf/2107.06463.pdf).
Haisheng Fu, Feng Liang, Jianping Lin, Bing Li, Mohammad Akbari, Jie Liang, Guohe Zhang, Dong Liu, Chengjie Tu, Jingning Han

## Paper Summary
Recently deep learning-based image compression methods have achieved significant achievements and gradually outperformed traditional approaches including the latest standard Versatile Video Coding (VVC) in both PSNR and MS-SSIM metrics. Two key components of learned image compression frameworks are the entropy model of the latent representations
and the encoding/decoding network architectures. Various models have been proposed, such as autoregressive, softmax, logistic mixture, Gaussian mixture, and Laplacian. Existing schemes only use one of these models. However, due to the vast diversity of images, it is not optimal to use one model for all images, even different regions of one image. In this paper, we propose a more flexible discretized Gaussian-Laplacian-Logistic mixture model (GLLMM) for the latent representations, which can adapt to different contents in different images and different regions of one image more accurately. Besides, in the encoding/decoding network design part, we propose a concatenated residual blocks
(CRB), where multiple residual blocks are serially connected with additional shortcut connections. The CRB can improve the learning ability of the network, which can further improve the compression performance. Experimental results using the Kodak and Tecnick datasets show that the proposed scheme outperforms all the state-of-the-art learning-based methods and existing compression standards including VVC intra coding (4:4:4 and 4:2:0) in terms of the PSNR and MS-SSIM.

### Environment 
* Python==3.6.4
* Tensorflow==1.14.0
* [RangeCoder](https://github.com/lucastheis/rangecoder)
```   
    pip3 install range-coder
```
* [Tensorflow-Compression](https://github.com/tensorflow/compression) ==1.2
```
    pip3 install tensorflow-compression or 
    pip3 install tensorflow_compression-1.2-cp36-cp36m-manylinux1_x86_64.whl
```
### Test Usage
* Download the pre-trained [models](https://drive.google.com/open?id=19b92ey1g30R2OvWupekLQNb3TjHs5HLX) (this model is optimized by MS-SSIM using lambda = 14) and unzip it.
* Put your images to the directory valid/ and run the py files
```
    python3 encoder.py
```
```
    python3 decoder.py
```
## Reconstructed Samples
Comparisons of reconstructed samples are given in the following.
![](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/figures/visualizationKodim21Ver2.png)
## Evaluation Results
![](https://github.com/ZhengxueCheng/Learned-Image-Compression-with-GMM-and-Attention/blob/master/figures/RD.PNG)
## Notes
This implementations are not original codes of our CVPR2020 paper, because original code is based on Tensorflow 1.9.0 and many features have been removed. This repo is a re-implementation, but the core codes are almost the same and results are also consistent with original results. This repo is also submitted to CVPR Workshop and Challenge on Leanred Image Challenge ([CLIC] (http://www.compression.cc/)) with the entry Kattolab in the Leaderboard.

If you think it is useful for your reseach, please cite our paper. Our original RD data in the paper is contained in the folder RDdata/.
```
@inproceedings{cheng2020image,
@inproceedings{cheng2020image,
    title={Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules},
    author={Cheng, Zhengxue and Sun, Heming and Takeuchi, Masaru and Katto, Jiro},
    booktitle= "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    year={2020}
}
