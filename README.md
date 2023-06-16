# A2S-USOD

## **Our new [A2S-v2 framework](https://github.com/moothes/A2S-v2) is accepted by CVPR 2023!**

In this work, we propose a new framework for Unsupervised Salient Object Detection (USOD) task.  
Details are illustrated in our paper: "[Activation to Saliency: Forming High-Quality Labels for Unsupervised Salient Object Detection](https://arxiv.org/abs/2112.03650)".  
Contact: [zhouhj26@mail2.sysu.edu.cn](zhouhj26@mail2.sysu.edu.cn).

## Update 2022/06/05
Code is available now!  
Our code is based on our [SOD benchmark](https://github.com/moothes/SALOD).  
Pretrained backbone: [MoCo-v2](https://github.com/facebookresearch/moco).  
Our trained weights: [Stage1-moco](https://drive.google.com/file/d/18Ne-48WeZL-SlpG0bE80f0p8zqkpeVpG/view?usp=sharing), [Stage1-sup](https://drive.google.com/file/d/1hFvNRYN7fJd2EvRHhuHzzH853tVBjdlJ/view?usp=sharing), [Stage2-moco](https://drive.google.com/file/d/1-9UpIjj4iXw35pKIDQdbIl2wqmbXFQYO/view?usp=sharing), [Stage2-sup](https://drive.google.com/file/d/1XS73VArH5yumaer0BLCJD7A_SsMztDCV/view?usp=sharing).  

Here we provide the generated saliency maps of our method in Google Drive: [Pseudo labels (Stage 1)](https://drive.google.com/file/d/1SaoX2EMUKn22lJtSQeQvCJUHjedrV3hR/view?usp=sharing) and [Saliency maps (Stage 2)](https://drive.google.com/file/d/1wQGDq7jBrzt5sqXgs7dM66iMga4H9n0b/view?usp=sharing), or download from [Baidu Disk](https://pan.baidu.com/s/1diqoo98ISjZs1smsL9t-RA) [g6xb].   


 ## Usage
 
 ```
 # Stage 1
 python3 train_stage1.py mnet --gpus=0 
 python3 test.py mnet --weight=path_to_weight --gpus=0 --crf --save
 # Copy the generated pseudo labels to dataset folder
 
 # Stage 2
 python3 train_stage2.py cornet --gpus=0
 python3 test.py cornet --weight=path_to_weight --gpus=0 [--save] [--crf]
 
 # To evaluate generated maps:
 python3 eval.py --pre_path=path_to_maps
 ```
 

## Results
![Result](https://github.com/moothes/A2S-USOD/blob/main/result.PNG)

Thanks for citing our work
```xml
@ARTICLE{zhou2023a2s1,
  title={Activation to Saliency: Forming High-Quality Labels for Unsupervised Salient Object Detection}, 
  author={Zhou, Huajun and Chen, Peijia and Yang, Lingxiao and Xie, Xiaohua and Lai, Jianhuang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2023},
  volume={33},
  number={2},
  pages={743-755},
  doi={10.1109/TCSVT.2022.3203595}}
```
