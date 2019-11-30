# Everybody Dance Now
This repository tries to implement [Everybody Dance Now](https://arxiv.org/abs/1808.07371) by pytorch.
Lot of inspiration has been taken from multiple repositories. Since I was encountering some problems running their repositories, I created a rectified version of theirs with more utilities and documentation for the purpose of learning. Please get back to me if faced with any issues or bugs.  
Repositories which I took inspiration from:-   
- [yanx27](https://github.com/CUHKSZ-TQL/EverybodyDanceNow_reproduce_pytorch) EverybodyDanceNow reproduced in pytorch
- [nyoki-pytorch](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow) pytorch-EverybodyDanceNow
- [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)

## Introduction
This project will enable the user to transfer whatever pose they want to the game character which otherwise involves a lot of finesse and experience. The network will detect those poses and will create a game character enacting those poses. In this project, I will only model and focus on generating poses for a single game character. In the paper Everybody dance now by Caroline Chan et.al[1], the group had focused on transferring the pose of a source video usually consisting of a professional dancer to a target video consisting of people with no experience in that field. I will extend it to game figures. I will also use the Openpose model and the Conditional GAN (CGAN) structure from pix2pix. However, I will not do multiple screen predictions by Generator as mentioned in paper for temporal coherence.

- Pose estimation
    - [x] Pose
    - [ ] Face
    - [ ] Hand
- [x] pix2pixHD
- [ ] FaceGAN
- [ ] Temporal smoothing

## Methodology
To solve the problem statement, we needed to convert a pose from domain f(x) to the desired game character in that pose g(x). Where x is the desired pose.  
The problem is broken into two parts:  
1. Pose estimation to procure x
2. Using CGAN model to transform x to desired game character pose

The pose estimation was successfully done by using the Openpose algorithm developed by CMU.  
![Alttext](https://raw.github.com/Vishwesh4/Everybody-dance-now/master/images/img1.png)

We have used the following architecture called pix2pix which is a type of CGAN developed by NVIDEA for the purpoe of transforming the image from one domain to another. Using this step we successfully obtain g(x).  

![Alttext](https://raw.github.com/Vishwesh4/Everybody-dance-now/master/images/img2.png)

## Results


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Note that your local environment should be cuda enabled from both pose detection and pose transfer.

### Prerequisites

Libraries needed

```
openpose
dominate
pytorch-gpu
```

### Dataset
### Training
### Testing
### Training with your Dataset
### Running the test files
Before running the main code ensure that the label images are in train_A folder and the target images are in train_B folder. A single image has been left for example.  
File description
* `1 - main codes/OpenPose_final_step1.ipynb` - Its the first step to transfer images to pose. The code is written for running on google collab due to additional packages that are required to install
* `2 - main codes/Main_Training_step2.ipynb` - This is 2nd step in pipeline. Main code for training. The results of the notebook will be found in checkpoints/Pose_final
* `3 - main codes/Main_Test_step3.ipynb` - This notebook is the 3rd step. It runs on cross validation set to give idea of losses and images. The results of the notebook will be found in checkpoints/Pose_test
* `4 - main codes/Pose_Transfer_step4.ipynb` - This notebook is the 4th step. This notebook finally transfers the given input poses to the target image. The results can be found in results folder
* `5 - main codes/for_epoch_pics.ipynb` - This notebook just displays iteration wise images for the purpose of visualization. The results can be found in checkpoints/Initial_epoch
* `6 - dataset` - directory for training. The poses goes to train_A folder and the target image goes to train_B folder
* `7 - dataset_test` - directory for cross validation set. The heirchacy is same as dataset folder.
* `8 - dataset_transfer` - Transfers pose mentioned in test_A directory to target images.
* `9 - config` - contains file for setting hyper parameters. See pix2pixHD repository for more information
* `10 - src` - directory containing pix2pixHD repository

## Reference
- [Everybody Dance Now](https://arxiv.org/abs/1808.07371 )
- [Image-to-Image Translation with Conditional Adversarial Networks(pix2pix)](https://arxiv.org/pdf/1611.07004.pdf)
