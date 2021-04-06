# Face Identity Disentanglement via Latent Space Mapping - Implement in pytorch with StyleGAN 2

## Description

Pytorch implementation of the paper *Face Identity Disentanglement via Latent Space Mapping* for both training and evaluation, with StyleGAN 2.
- Paper: https://arxiv.org/abs/2005.07728
- Official TensorFlow Implementation: https://github.com/YotamNitzan/ID-disentanglement
- Important: this implementation does not follow the exact same architecture of the original paper. See changes below

## Changes from original paper
- instead of using a Discriminator loss for the mapper. We have used several other losses such as:
    - LPIPS Loss (The Unreasonable Effectiveness of Deep Features as a Perceptual Metric, Zhang el al, 2018)
    - MSE Loss
    - Different ID Loss
    - Different landmark detector
- The reason for those changes resides in the fact that the training procedure with Discriminator is often
hard and does not converge. We have found that replacing the Discriminator with LPIPS and MSE losses
  we can achieve the same result. Nevertheless, our code supports training with a discriminator which can be
  activated using the configuration.
- The other changes are due to better Recognition models that have developed since the original paper was published

## Setup

We used several **pretrained models**: 
- StyleGan2 Generator for image size 256 - 550000.pt
- ID Encoder - model_ir_se50.pth
- Landmarks Detection - mobilefacenet_model_best.pth.tar

Weight files attached at this [Drive folder](https://drive.google.com/drive/folders/18K5YBBJRiCIradtttlLcdtSyLUo3cUI5?usp=sharing).

You can also find at the above link our **environment.yml** file to create a relevant conda environment.

## Training

The dataset is comprised of StyleGAN 2 generated images and W latent codes. see Utils/**data_creator.py**.

Examples of our generated dataset attached at this [Drive folder](https://drive.google.com/drive/folders/1SW7fE9KQV8XXYeluB3MavuAWlObwq65J?usp=sharing).

To train the model run **train_script.py**, you can change parameters in **Configs/** folder.

## Inference

Try **Inference.ipynb** notebook to disentangle identity from attributes by yourself

## Checkpoints

Our pretrained checkpoint attached at this [Drive folder](https://drive.google.com/drive/folders/1Z7BTqSrPi37I4mH6C7RCJNr2v6Zx69pB?usp=sharing).

## Results

![Results](https://github.com/danielroich/ID-disentanglement-Pytorch/blob/master/Results.jpg)
