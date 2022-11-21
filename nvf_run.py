import gpt_2_simple as gpt2 # This line should be in the first line. Otherwise, there maybe Aborted (core dumped)
from ast import arg
import sys, os
import matplotlib.pyplot as plt
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), 'models'))
import numpy as np
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
import argparse
from math import log
from nvf import NVFTabular, NVFToy, NVFVision
import yaml
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Flow trainer")
parser.add_argument("--train_batch", default=32, type=int, help="batch size")
parser.add_argument("--test_batch", default=1, type=int, help="batch size")
parser.add_argument("--epoch", default=1000, type=int, help="maximum iterations")
parser.add_argument("--hidden_features", default=32, type=int, help="n feature")
parser.add_argument("--residual_blocks", default=1, type=int, help="residual_blocks")
parser.add_argument("--num_bins", default=4, type=int, help="num_bins")
parser.add_argument("--anneal_learning_rate", default=0, type=bool, help="anneal_learning_rate")
parser.add_argument("--n_flow", default=20, type=int, help="number of flows in each block")
parser.add_argument("--K", default=10, type=int, help="number of clusters")
parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
parser.add_argument("--dropout", default=0.2, type=float, help="learning rate")
parser.add_argument("--data", default='miniboone', type=str, help="data")
parser.add_argument("--is_test", default=0, type=bool, help="is_test")
parser.add_argument("--use_wandb", default=0, type=bool, help="use wandb")
parser.add_argument("--continue_iter", default=0, type=int, help="continue_iter")

parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--embed_dim", default=16, type=int, help="dimension of embeddings")
parser.add_argument("--reschannel", default=64, type=int, help="number of reschannel")
parser.add_argument("--resblock", default=1, type=int, help="number of resblock")
parser.add_argument("--latentchannel", default=64, type=int, help="number of latentchannel")
parser.add_argument("--n_embed", default=16, type=int, help="number of embeddings")
parser.add_argument("--no_lu", action="store_true", help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--gpt_lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=32, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--data_name", default="cifar10", type=str, help="dataset name")
parser.add_argument("--extract_code_steps", type=int, default=10000)
parser.add_argument("--gpt_temp", default=0.7, type=float, help="temperature of gpt")
parser.add_argument("--noise_temp", default=0.7, type=float, help="temperature of noise")
parser.add_argument("--pretrained", type=str, default="checkpoints/model_mle_030001.pt")
parser.add_argument("--latent_code_path", type=str, default="./val/", help="path that you want to save generated imgs")
parser.add_argument("--sample_index", type=int, default=2, help="the identifier for the generated image")
parser.add_argument("--txt_path", type=str, default="./output/", help="the path of generated txt (latent code)")
parser.add_argument("--run_name", type=str, default="gpt-2-run-124M-flow", help="the name of gpt2 model, same as the name in train_gpt2.py")
parser.add_argument("--startFile", type=str, default='')
parser.add_argument("--width", type=int, default=16, help="image latent width")
parser.add_argument("--height", type=int, default=16, help="image latent height")
parser.add_argument("--num_imgs", type=int, default=16, help="number of images that you want to generate")
parser.add_argument("--num_vocab", type=int, default=512, help="number of vocabs")
parser.add_argument("--max_num_attempts", type=int, default=6, help="maximum number of attempts")
parser.add_argument("--test_interval", type=int, default=100, help="interval for testing")
parser.add_argument("--save_interval", type=int, default=10000, help="interval for saving checkpoints")

eps = 1e-10
args = parser.parse_args()
print(args)
K = args.K
use_wandb = args.use_wandb

if use_wandb:
    import wandb
    wandb.init(project="NormalizingFlow",name = args.data+'-mixture',config = args)

trainset, valset, testset, data_type = utils.io.get_data(args.data)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=0)

num_layers = args.n_flow

if data_type ==  "tabular":
    model = NVFTabular(trainset.dim, K, num_layers, args.residual_blocks, args.hidden_features, args.num_bins, args.dropout)
elif data_type == "toy":
    model = NVFToy(K, num_layers, residual_blocks=args.residual_blocks, hidden_features=args.hidden_features)
elif data_type == "vision":
    model = NVFVision(args, device)

if data_type == "vision":
    optimizer = optim.Adam(model.model.parameters(), lr=args.lr)

    if args.is_test:
        model.model.load_state_dict(torch.load(f"./checkpoints/model_010001_5bits16vocab16embeddim.pt"))
    else:
        ### Step1: Likelihood training for NF train_condition.py ###
        model.train_condition(trainloader, optimizer)

    ### Step2: VQ step for images extract_code.py ###
    model.extract_code(trainloader, is_train = True)
    model.extract_code(testloader, is_train = False)
    
    ### Step3: Prior training with GPT train_gpt2.py ###
    session = gpt2.start_tf_sess()
    gpt2.finetune(session,
            batch_size=args.test_batch,
            dataset=model.args.dataset,
            model_name='124M',
            steps=args.iter,
            restore_from='latest',
            run_name=model.args.run_name,
            print_every=10,
            sample_every=1000,
            multi_gpu=False,
            save_every=1000,
            learning_rate=args.gpt_lr,
        )

    ### Step4: Generation and evaluation with both GPT and NF. - generate.py ###
    model.gpt2_generate()
    txt_lists = model.getFileName()

    # Iteratively generate images.
    for i in range(1):
        model.text_to_image(txt_lists[i*16:i*16+16], img_id = args.sample_index + i)

    ### Step 5: Evaluation. test_condition.py and test_gpt2.py
    model.test_condition(testloader)

    dataset = f"./output/latent_codes/test_images_{args.n_bits}bits{args.n_embed}vocab{args.embed_dim}embeddim.txt"
    session = gpt2.start_tf_sess()
    gpt2.finetune(session,
            batch_size=args.test_batch,
            dataset=dataset,
            model_name='124M',
            steps=args.iter,
            restore_from='latest',
            run_name=run_name,
            print_every=10,
            sample_every=500,
            multi_gpu=False,
            save_every=500,
            learning_rate=0.0,
        )
else:
    flow = model.flow
    model0 = model.phi

    if args.continue_iter>0:
        path_flow = f"checkpoints/flow_{args.data}_{str(args.continue_iter).zfill(6)}.pt"
        path_phi = f"checkpoints/posterior_{args.data}_{str(args.continue_iter).zfill(6)}.pt"
        model.load_flow(path_flow)
        model.load_phi(path_phi)

    if args.is_test:
        flow.eval()
        val_loss = model.test(valloader)
        test_loss = model.test(testloader)
        print("val loss:", val_loss)
        print("test loss:", test_loss)
    else:
        optimizer = optim.Adam(list(flow.parameters())+list(model0.parameters()),lr=args.lr)
        batchperepoch = int(np.ceil(trainset.data.shape[0] / args.train_batch))
        if args.anneal_learning_rate:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch*batchperepoch, 0)
        else:
            scheduler = None

        log_1_K = log(1/K)
        i = 0
        P = []

        for j in range(args.epoch):
            for x in trainloader:
                flow.train()
                if args.anneal_learning_rate:
                    scheduler.step(i)
                x = x.cuda()
                loss,loss_kl = model.nll(x,return_kl=True)
                loss_full = loss.mean() + loss_kl.mean()
                loss_full.backward()
                clip_grad_norm_(flow.parameters(), 5)
                optimizer.step()
                i+=1
                if (i + 1) % 100 == 0:
                    val_loss = model.test(valloader)
                    test_loss = model.test(testloader)
                    log_dict = {
                    "iteration":i+1,
                    "train loss": loss.item(),
                    "train loss (kl)": loss_kl.item(),
                    "val nll": val_loss,
                    "test nll": test_loss,
                    }
                    if use_wandb:
                        wandb.log(log_dict)
                    print(yaml.dump(log_dict))
                    torch.save(
                        flow.state_dict(), f"checkpoints/flow_{args.data}_{str(i + 1).zfill(6)}.pt"
                    )
                    torch.save(
                        model0.state_dict(), f"checkpoints/posterior_{args.data}_{str(i + 1).zfill(6)}.pt"
                    )

                    if data_type == "toy":
                        utils.io.savefig_toy(flow,K,i)