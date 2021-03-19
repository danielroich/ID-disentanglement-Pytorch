# Face Identity Disentanglement via Latent Space Mapping - Implement in pytorch with StyleGAN 2

## Description

Pytorch implementation of the paper *Face Identity Disentanglement via Latent Space Mapping* for both training and evaluation, with StyleGAN 2.
- Paper: https://arxiv.org/abs/2005.07728
- Official TensorFlow Implementation: https://github.com/YotamNitzan/ID-disentanglement

## Setup

We used several **pretrained models**: 
- StyleGan2 Generator for image size 256 - 550000.pt
- ID Encoder - model_ir_se50.pth
- Landmarks Detection - mobilefacenet_model_best.pth.tar

Weight files attached at this [Drive folder](https://drive.google.com/drive/folders/18K5YBBJRiCIradtttlLcdtSyLUo3cUI5?usp=sharing).

You can also find at the above link our **environment.yml** file to create a relevant conda environment.

## Training

The dataset is comprised of StyleGAN 2 generated images and W latent codes.

Our generated dataset attached at this [Drive folder](https://drive.google.com/drive/folders/1SW7fE9KQV8XXYeluB3MavuAWlObwq65J?usp=sharing).

To train the model run **Training_Notebook.py**, you can change parameters in **Configs/** folder.

## Inference



## Results



## Checkpoints
Our pretrained checkpoint attached at this [Drive folder](https://drive.google.com/drive/folders/1Z7BTqSrPi37I4mH6C7RCJNr2v6Zx69pB?usp=sharing).
