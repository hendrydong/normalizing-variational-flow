# NVF

> This repository contains the codes for [Normalizing Flow with Variational Latent Representation](https://arxiv.org/abs/2211.11638)  
by Hanze Dong\*, Shizhe Diao\*, Weizhong Zhang, Tong Zhang.

## An one-line pipeline
Take vision data as an example, by simply running the following command, the whole pipeline will be executed stage by stage,

`source install.sh;cd vision;bash run.sh`

which will run environment installation, dataset prepration, training and testing one by one.
The details of each stage are illustrated as follows.

## Setup
run `source install.sh` to install all the dependencies, download datasets, and set up the enviroment.

## Data Preparation

In our experiments, the data are placed at `./data`. 

For toy data, they are synthetic, which do not need storage.

For tabular data, you need to download them at https://zenodo.org/record/1161203#.Wmtf_XVl8eN. For example, we placed `miniboone` in the `./data` for your reference.

For vision data, we use `torchvision` as `Dataloader`. You may refer to https://pytorch.org/vision/stable/datasets.html. For most dataset, it can be download automatically with the `torchvision`.


## Training and Evaluation

We have 3 parts for our density estimation experiments:

1. Toy data: 2d toy data, such as moon, Gaussian mixture datasets. The main difference for NVF is to incorperate the latent variable to encode different clusters, which simplify the transport function significantly. For this part we use Jupyter Notebook for better illustration.

2. Tabular data: nd vector data (POWER, GAS, HEPMASS, MINIBOONE, BSDS300). The latent variable is designed as discrete varibles, implemented as K-dim one-hot vector. The posterior estimation is done by gumble-softmax.

3. Vision data: image data (MNIST, FashionMNIST, CIFAR, CelebA). The latent variable is designed as a sequence, where the posterior estimation is done by vector quantization, the prior estimation is modelled by transformer.

### Toy data

```
python nvf_run.py --epoch 100000 --batch 1024  --residual_blocks 1  --lr 1e-3  --K 2 --data moons --n_flow 10 --hidden_features 16
```


### Tabular data

Start training with
```
python nvf_run.py --epoch 1000 --batch 128 --hidden_features 32 --residual_blocks 1 --data miniboone --dropout 0.2 --lr 5e-4 --num_bins 4 --K 10
```
where we take `miniboone` for example.

The evaluation is done by
```
python nvf_run.py --epoch 1000 --batch 128 --hidden_features 32 --residual_blocks 1 --data miniboone --dropout 0.2 --lr 5e-4 --num_bins 4 --K 10 --is_train 1 --continue_iter 35000 
```
where the `test_iter` indicate the best validation iteration.

### Vision Data

```CUDA_VISIBLE_DEVICES=0 python nvf_run.py --data mnist --n_flow 32 --lr 1e-5 --epoch 1 --is_test 0```

## Loss computation

Our loss contains two parts, the reconstruction loss and MLE loss.

```
Reconstruction loss:
diff_b,diff_t = model(image, z_list)
diff_b: latent VQ error
diff_t: reconstruction loss of image
```
```
MLE loss:
log_p, logdet, z_outs = model.module.dec(image,conditions)
loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
```

Note that in calc_loss:
`loss = - log_likelihood / (log(2) * n_pixel)`

Thus, in the Full model 
`NLL_Full = loss + loss_GPT / (log(2) * n_pixel)`


## Acknowledgements

Our codes was gratefully forked from https://github.com/bayesiains/nflows.


## Contact
For help or issues, as well as even personal communication related to this repo, please contact:

Hanze Dong: A (AT) B, where A=hdongaj, B=ust.hk.
Shizhe Diao: A (AT) B, where A=sdiaoaa, B=ust.hk.


## Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@article{nvf2022,
  title={Normalizing Flow with Variational Latent Representation},
  author={Dong, Hanze and Diao, Shizhe and Zhang, Weizhong and Zhang, Tong},
  journal={arXiv preprint arXiv:2211.11638},
  year={2022}
}
```
