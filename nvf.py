import torch
from torch import nn 
from models.nflows.flows.base import Flow
from models.nflows.distributions.normal import StandardNormal,ConditionalDiagonalNormal
from models.nflows.transforms.base import CompositeTransform
from models.nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from models.nflows.transforms import PiecewiseRationalQuadraticCouplingTransform,MaskedPiecewiseRationalQuadraticAutoregressiveTransform,LULinear,RandomPermutation,AffineCouplingTransform
from models.nflows.transforms.permutations import ReversePermutation
from math import log
from models.nflows.nn.nets import ResidualNet
import utils
import torchvision
import os
import time
import gpt_2_simple as gpt2
from tqdm import tqdm
import numpy as np
from models.model import QAE_v2

class NVF:
    def __init__(**kwargs):
        pass

    def nll(self,x, return_kl = False):
        logits = self.phi(x)
        y_oh = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=True)
        p = torch.nn.functional.softmax(logits,1)
        loss_kl = torch.sum(p*torch.log(p+self.eps),1)-log(1/self.K) 
        loss_kl = loss_kl.mean()
        loss = -self.flow.log_prob(inputs=x, context = y_oh).mean()
        if return_kl:
            return loss,loss_kl
        return loss

    def load_flow(self,path):
        self.flow.load_state_dict(torch.load(path))

    def load_phi(self,path):
        self.phi.load_state_dict(torch.load(path))

    def test(self, testloader, max_iter=10):
        k = 1
        loss = 0
        lossk = 0
        with torch.no_grad():
            for x_te in testloader:
                x_te = x_te.cuda()
                LOSS_K = []
                for k_cluster in range(self.K):
                    y_oh = torch.zeros(x_te.shape[0],self.K).float().cuda()
                    y_oh[:,k_cluster] = 1
                    lk = self.flow.log_prob(inputs=x_te,context = y_oh)
                    LOSS_K.append(lk.reshape(-1,1))  
                LOSS_K = torch.cat(LOSS_K,1) # B x K log scale
                LOSS_upper = torch.max(LOSS_K,1)[0].reshape(-1,1) # B x 1 log scale
                LOSS_K -= LOSS_upper 
                LOSS_K = torch.exp(LOSS_K) # density scale
                for k_cluster in range(self.K):
                    LOSS_K[:,k_cluster] *= 1/self.K
                
                LOSS_K = torch.log(torch.sum(LOSS_K,1)).mean().item()
                lossk += -LOSS_K
                loss += -LOSS_K - LOSS_upper.mean().item()
                if k==max_iter:
                    break
                k+=1
            if k<max_iter:
                k-=1
        return loss/k

class NVFTabular(NVF):
    def __init__(self, dim, K, num_layers, residual_blocks, hidden_features, num_bins, dropout, inference_hidden=256, eps = 1e-10):
        transforms = []
        base_dist = StandardNormal(shape=[dim])

        for _ in range(num_layers):
            transforms.append(RandomPermutation(features=dim))         
            transforms.append(LULinear(dim, identity_init=True))
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(            
                    features=dim,
                    hidden_features=hidden_features,
                    context_features=K,
                    num_bins=num_bins,
                    tails='linear',
                    tail_bound=3,
                    num_blocks=residual_blocks,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=torch.relu,
                    dropout_probability=dropout,
                    use_batch_norm=0))
        transforms.append(RandomPermutation(features=dim))         
        transforms.append(LULinear(dim, identity_init=True))
        transform = CompositeTransform(transforms)

        self.phi = nn.Sequential(
        nn.Linear(dim,inference_hidden),
        nn.ReLU(),
        nn.Linear(inference_hidden,inference_hidden),
        nn.ReLU(),
        nn.Linear(inference_hidden,K)).cuda()

        self.flow = Flow(transform, base_dist).cuda()
    
        self.eps = eps
        self.K = K

class NVFToy(NVF):
    '''
    2d dataset
    '''
    def __init__(self, K, num_layers, residual_blocks=1, hidden_features=4, inference_hidden=16, eps = 1e-10):
        m = hidden_features
        base_dist = StandardNormal(shape=[2])

        transforms = []
        for i in range(num_layers):
            transforms.append(ReversePermutation(features=2))
            transforms.append(LULinear(2, identity_init=True))
            transforms.append(AffineCouplingTransform(
                    mask=utils.create_alternating_binary_mask(features=2, even=(i % 2 == 0)),
                    transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        hidden_features=m,
                        num_blocks=residual_blocks,
                        use_batch_norm=False,
                        context_features=K
                    )))
        transforms.append(ReversePermutation(features=2))
        transforms.append(LULinear(2, identity_init=True))
        transform = CompositeTransform(transforms)
        self.flow = Flow(transform, base_dist).cuda()
        self.eps = eps
        self.K = K
        self.phi = nn.Sequential(
                nn.Linear(2,inference_hidden),
                nn.ReLU(),
                nn.Linear(inference_hidden,inference_hidden),
                nn.ReLU(),
                nn.Linear(inference_hidden,2)).cuda()


class NVFVision:
    def __init__(self, args, device):
        self.device = device
        self.args = args
        self.args.dataset = f"./output/latent_codes/train_images_{args.n_bits}bits{args.n_embed}vocab{args.embed_dim}embeddim.txt"
        self.args.run_name = f'gpt-2-run-124M-flow-{args.n_bits}bits{args.n_embed}vocab{args.embed_dim}embeddim'

        self.args.txt_path = args.txt_path + str(args.gpt_temp) + "_" + str(args.noise_temp) + "/"
        if not os.path.exists(self.args.txt_path):
            os.mkdir(self.args.txt_path)
        # print("txt_path = ", self.args.txt_path)

        self.model_single = QAE_v2(
            in_channel=3,
            channel=args.latentchannel,
            n_res_block=args.resblock,
            n_res_channel=args.reschannel,
            embed_dim=args.embed_dim,
            n_embed=args.n_embed,
            decay=0.99,
            n_flow=args.n_flow,
            n_block=args.n_block,
            affine=args.affine,
            conv_lu=not args.no_lu
        )
        self.model = nn.DataParallel(self.model_single)
        self.model = self.model.to(device)
        self.n_bins = 2.0 ** self.args.n_bits

        self.z_sample = []
        z_shapes = self.calc_z_shapes(3, self.args.img_size, self.args.n_flow, self.args.n_block)
        for z in z_shapes:
            z_new = torch.randn(self.args.train_batch, *z) * self.args.temp
            self.z_sample.append(z_new.to(self.device))

        self.latent_code_root_dir = "./output/latent_codes"
        if not os.path.exists(self.latent_code_root_dir):
            os.makedirs(self.latent_code_root_dir)

        if args.use_wandb:
            import wandb
            wandb.init(project="NormalizingFlow",name = args.data+'-mixture',config = args)

    def blankLines(self, height):
        lines = []
        for i in range(0, height):
            lines.append('')

        return lines

    def readFile(self, filepath):
        with open(filepath) as f:
            content = f.read()

        return content

    def getFileName(self):
        f_list = os.listdir(self.args.txt_path)
        required_list = []
        for f_name in f_list:
            if os.path.splitext(f_name)[1] == '.txt':
                required_list.append(f_name)
        return required_list

    def gpt2_generate(self):
        width = self.args.width
        height = self.args.height
        startFile = self.args.startFile
        run_name = self.args.run_name
        succ_num_imgs = 0
        sess = gpt2.start_tf_sess()
        
        for ii in range(0, self.args.num_imgs):
            sess = gpt2.reset_session(sess)
            gpt2.load_gpt2(sess, run_name=run_name)

            debug = []
            lines = self.blankLines(height)

            prefix = '15u'
            first = True
            format_resolution = True

            cycleCount = 0

            while True:
                debug.append('\n\ncycle %i' % cycleCount)
                debug.append('prefix:')
                debug.append(prefix)
                cycleCount += 1
                print("cycleCount = ", cycleCount, " / max_try = ", self.args.max_num_attempts)
                if(cycleCount > self.args.max_num_attempts):
                    #if loops too long for only one figure, directly stop program.
                    return 1

                if startFile and first:
                    text = self.readFile(startFile)
                else:
                    text = gpt2.generate(sess, run_name=run_name, prefix=prefix, temperature=self.args.gpt_temp, return_as_list=True)[0]
                first = False
                format_resolution = True

                debug.append('output:')
                debug.append(text)
                print('\n\noutput:')
                print(text)

                newLines = text.split('\n')

                direction = None
                lastIndex = None
                for line in newLines:
                        split = line.split(' ')[:width + 2]
                        marker = split[0]
                        if len(marker) == 3:
                            try:
                                index = int(marker[0:2])
                            except:
                                break

                            if marker[2] == 'u':
                                continue

                            if direction == None:
                                direction = marker[2]

                            if marker[2] != direction:
                                debug.append('direction changed')
                                print('direction changed')
                                break

                            if lastIndex != None:
                                print("split", split)
                                if marker[2] == 'd' and index <= lastIndex:
                                    debug.append('bad line order')
                                    print('bad line order')
                                    break
                                elif marker[2] == 'u' and index >= lastIndex:
                                    debug.append('bad line order')
                                    print('bad line order')
                                    break
                            lastIndex = index

                            split[0] = marker.replace('u', 'd')

                            if len(split) > self.args.width+1:  #1 denotes 00d
                                print("BREAK! exceeded width! ", split)
                                split = split[:self.args.width+1]
                                # format_resolution = False
                                # break
                            if len(split) < self.args.width+1:
                                print("BREAK! less than width! ", split)
                                # format_resolution = False
                                # break

                            try:
                                lines[index] = ' '.join(split)
                            except IndexError:
                                debug.append('line number out of range')
                                print('line number out of range')
                                break

                if format_resolution == False:
                    continue

                topIndex = None
                for i in range(0, height):
                    if lines[i]:
                        topIndex = i
                        break
                if topIndex == None:
                    print("topIndex == None")
                    continue

                bottomIndex = None
                for i in range(topIndex, height):
                    if lines[i]:
                        bottomIndex = i
                    else:
                        break

                debug.append('top %i bottom %i' % (topIndex, bottomIndex))
                print('\n\ntop %i bottom %i' % (topIndex, bottomIndex))

                sectionSize = 5
                if topIndex > 0:
                    section = lines[topIndex:min(topIndex+sectionSize+1, bottomIndex+1)]
                    section.reverse()
                    for i in range(0, len(section)):
                        section[i] = section[i].replace('d', 'u')

                elif bottomIndex < height - 1:
                    section = lines[max(bottomIndex-sectionSize, topIndex):bottomIndex+1]

                else:
                    filename = '%03d-%i' % (self.args.gpt_temp, int(time.time()))

                    text_file = open(f'{self.args.txt_path}/%s.txt' % filename, 'w')
                    text_file.write('\n'.join(lines))
                    text_file.close()

                    debug_file = open(f'{self.args.txt_path}/%s.log' % filename, 'w')
                    debug_file.write('\n'.join(debug))
                    debug_file.close()

                    succ_num_imgs += 1
                    print('saved ! succ_num_imgs = ', succ_num_imgs, "/", ii+1)
                    break

                prefix = '\n'.join(section)

    def calc_z_shapes(self, n_channel, input_size, n_flow, n_block):
        z_shapes = []

        for i in range(n_block - 1):
            input_size //= 2
            n_channel *= 2
            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes


    def calc_loss(self, log_p, logdet, image_size, n_bins):
        n_pixel = image_size * image_size * 3

        loss = -log(n_bins) * n_pixel
        loss = loss + logdet + log_p

        return (
            (-loss / (log(2) * n_pixel)).mean(),
            (log_p / (log(2) * n_pixel)).mean(),
            (logdet / (log(2) * n_pixel)).mean(),
        )

    def train_condition(self, dataloader, optimizer):
        step_per_epoch = len(dataloader)
        for epoch in range(self.args.epoch):
            for step, (image, labels) in enumerate(dataloader):
                if image.shape[0] != self.args.train_batch:
                    continue
                image = image.to(self.device)
                image = image * 255

                if self.args.n_bits < 8:
                    image = torch.floor(image / 2 ** (8 - self.args.n_bits))

                image = image / self.n_bins - 0.5
                quant_b, diff_b, id_b = self.model.module.encode(image)
                condition1 = self.model.module.enc_1(quant_b)
                condition2 = self.model.module.enc_2(quant_b)
                condition3 = self.model.module.enc_3(quant_b)
                condition4 = self.model.module.enc_4(quant_b)
                conditions = [condition4,condition3,condition2,condition1]

                log_p, logdet, z_outs = self.model.module.dec(image,conditions)

                logdet = logdet.mean()
                loss, log_p, log_det = self.calc_loss(log_p, logdet, self.args.img_size, self.n_bins)    

                loss_full = loss + diff_b.mean()
                
                self.model.zero_grad()
                loss_full.backward()
                # warmup_lr = self.args.lr * min(1, i * batch_size / (50000 * 10))
                warmup_lr = self.args.lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()

                print(f"Epoch [{epoch}]; Step [{step} / {step_per_epoch}]; Loss: {loss.item():.5f}; lossFull: {loss_full.item():.5f};  log_p: {log_p:.7f}")
                if self.args.use_wandb:
                    wandb.log({"loss": loss.item(), "lossFull": loss_full.item(), "log_p": log_p})

                global_step = (epoch + 1) * step
                if step % self.args.test_interval == 0:
                    with torch.no_grad():
                        self.model_single.rec(image, self.z_sample).cpu().data
                        
                        torchvision.utils.save_image(
                            self.model_single.rec(image, self.z_sample).cpu().data,
                            f"sample/{str(global_step + 1).zfill(6)}.png",
                            normalize=True,
                            nrow=10,
                            range=(-0.5, 0.5),
                        )

                if step % self.args.save_interval == 0:
                    torch.save(
                        self.model.state_dict(), f"checkpoints/model_{str(global_step + 1).zfill(6)}_{self.args.n_bits}bits{self.args.n_embed}vocab{self.args.embed_dim}embeddim.pt"
                    )
                    torch.save(
                        optimizer.state_dict(), f"checkpoints/optim_{str(global_step + 1).zfill(6)}_{self.args.n_bits}bits{self.args.n_embed}vocab{self.args.embed_dim}embeddim.pt"
                    )

    def extract_code(self, dataloader, is_train):
        if is_train: 
            latent_code_path = f"{self.latent_code_root_dir}/train_images_{self.args.n_bits}bits{self.args.n_embed}vocab{self.args.embed_dim}embeddim.txt"
        else:
            latent_code_path = f"{self.latent_code_root_dir}/test_images_{self.args.n_bits}bits{self.args.n_embed}vocab{self.args.embed_dim}embeddim.txt"
        f_write = open(latent_code_path, "w")

        for image, labels in tqdm(dataloader):
            image = image.to(self.device)
            image = image * 255

            if self.args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - self.args.n_bits))

            image = image / self.n_bins - 0.5
            image0 = image  # [16, 3, 64, 64]

            with torch.no_grad():
                _, _, id_b = self.model_single.encode(image0)
                id_b = id_b.detach().cpu().numpy()
                bsz = id_b.shape[0]
                width = id_b.shape[1]
                height = id_b.shape[2]
                lines = []

                for i in range(bsz):
                    for x in range(height):
                        split = ['%02dd' % x]

                        for y in range(width):
                            color = id_b[i, x, y]
                            s = str(color)
                            split.append(s)
                        lines.append(' '.join(split))

                    reversed = []
                    for line in lines:
                        reversed.insert(0, (line.replace('d ', 'u ', 1)))
                    f_write.writelines('\n'.join(reversed))
                    f_write.write('\n')
                    f_write.writelines('\n'.join(lines))
                    f_write.write('\n')
                    lines = []
        f_write.close()

    def text_to_image(self, txt_list, img_id):
        batch_pixels = []
        for i_txt, path in enumerate(txt_list):
            if i_txt >= self.args.test_batch:
                break
            print("path: ", path)
            text = self.readFile(self.args.txt_path + path)
            lines = text.split('\n')
            img_pixels = []
            for line in lines:
                line_pixels = []
                split = line.split(' ')

                marker = split[0]
                if len(marker) == 3:
                    index = int(marker[0:2])

                    for x in range(len(split) - 1):
                        s = split[x + 1]
                        if len(s) == 0 or len(s) > 3:
                            print("LENGTH of s is not expected! s: ", s, "marker: ", marker)
                            s = "0"

                        try:
                            pixel = int(s)
                        except:
                            print("error when transfer!")
                            pixel=31
                        if(pixel < 0 or pixel >= self.args.num_vocab):
                            print("exceeded vocab! line_pixels = ", line_pixels, "s = ", s)
                            pixel = self.args.num_vocab - 1

                        line_pixels.append(pixel)

                    if len(line_pixels) > self.args.width:
                        print("exceeded width! ", line_pixels, "path = ", path)
                        line_pixels = line_pixels[0:self.args.width]
                    while len(line_pixels) < self.args.width:
                        print("less than width! ", line_pixels, "path = ", path)
                        line_pixels.append(self.args.num_vocab - 1)

                img_pixels.append(line_pixels)
            if(len(img_pixels) > self.args.height):
                continue
            batch_pixels.append(img_pixels)

        batch_pixels = torch.LongTensor(batch_pixels).to(self.device)  #torch.Size([10, 16, 16])
        print("batch_pixels.shape = ", batch_pixels.shape)

        z_shapes = self.calc_z_shapes(3, self.args.img_size, self.args.n_flow, self.args.n_block)
        z_sample = []

        for z in z_shapes:
            z_new = torch.randn(self.args.test_batch, *z) * self.args.noise_temp
            z_sample.append(z_new.to(self.device))

        torchvision.utils.save_image(
            self.model_single.decode_code(batch_pixels, z_sample).cpu().data,
            self.args.latent_code_path + '%.1f-%.1f.png' % (self.args.gpt_temp, self.args.noise_temp),
            normalize=True,
            nrow=4,
            range=(-0.5, 0.5),
        )

    def test_condition(self, dataloader):
        z_shapes = self.calc_z_shapes(3, self.args.img_size, self.args.n_flow, self.args.n_block)

        loss = 0
        loss0 = 0
        i = 0
        print('start to test')
        loss_list = []
        for image, labels in dataloader:
            image = image.to(self.device)

            image = image * 255

            if self.args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - self.args.n_bits))

            image = image / self.n_bins - 0.5
            image0 = image
            quant_b, diff_b, id_b = self.model.module.encode(image)
            condition1 = self.model.module.enc_1(quant_b)
            condition2 = self.model.module.enc_2(quant_b)
            condition3 = self.model.module.enc_3(quant_b)
            condition4 = self.model.module.enc_4(quant_b)
            conditions = [condition4,condition3,condition2,condition1]
            log_p, logdet, z_outs = self.model.module.dec(image,conditions)

            logdet = logdet.mean()
            loss, log_p, log_det = self.calc_loss(log_p, logdet, self.args.img_size, self.n_bins)    
            print('NLL loss:',loss.mean().item())
            loss_list.append(loss.mean().item())
            loss = 0
            loss0 = 0
            for j in range(10):
                z_sample = []
                for z in z_shapes:
                    z_new = torch.randn(self.args.train_batch, *z) * self.args.temp
                    z_sample.append(z_new.to(self.device))
                with torch.no_grad():
                    diff_b,diff_t = self.model(image0 , z_sample)
                    loss+=diff_b.mean()
                    loss0+=diff_t.mean()
                    torchvision.utils.save_image(
                        self.model_single.rec(image0,z_sample).cpu().data,
                        f"sample_test/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=4,
                        range=(-0.5, 0.5),
                    )
                    i+=1
                
            print(loss.item())
            print(loss0.item())

        print("average nll:", np.mean(loss_list))
