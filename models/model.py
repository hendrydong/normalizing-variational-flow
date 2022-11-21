import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from torch.nn import Parameter
import distributed as dist_fn
logabs = lambda x: torch.log(torch.abs(x))
from models.model_base import Encoder as Encoder_base
from models.model_base import Decoder as Decoder_base


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())
        #self.q_param = torch.nn.ParameterList([self.embed,self.cluster_size,self.embed_avg])

    def forward(self, input):
        flatten = input.reshape(-1, self.dim) 

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class SpectralNorm2d(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm2d, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        self.u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        self.v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        self.u.data = self._l2normalize(self.u.data)
        self.v.data = self._l2normalize(self.v.data)
        self.w_bar = Parameter(w.data)
        del self.module._parameters[self.name]

    def _l2normalize(self, x, eps=1e-12):
        r"""Function to calculate the ``L2 Normalized`` form of a Tensor
        Args:
            x (torch.Tensor): Tensor which needs to be normalized.
            eps (float, optional): A small value needed to avoid infinite values.
        Returns:
            Normalized form of the tensor ``x``.
        """
        return x / (torch.norm(x) + eps)

    def forward(self, *args):
        r"""Computes the output of the ``module`` and appies spectral normalization to the
        ``name`` attribute of the ``module``.

        Returns:
            The output of the ``module``.
        """
        height = self.w_bar.data.shape[0]
        for _ in range(self.power_iterations):
            self.v.data = self._l2normalize(
                torch.mv(torch.t(self.w_bar.view(height, -1)), self.u)
            )
            self.u.data = self._l2normalize(
                torch.mv(self.w_bar.view(height, -1), self.v)
            )
        sigma = self.u.dot(self.w_bar.view(height, -1).mv(self.v))
        setattr(self.module, self.name, self.w_bar / sigma.expand_as(self.w_bar))
        return self.module.forward(*args)


class BlockSN(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(in_channel, channel, 3, padding=1)),
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(channel, in_channel, 1)),
        )
    def forward(self, input):
        out = self.conv(input)
        return out  

class ResBlockSN(nn.Module):
    def __init__(self, channels, res_channel, n_blocks):
        super().__init__()
        block = [BlockSN(channels,res_channel) for i in range(n_blocks)]
        self.block = nn.Sequential(*block)
        #self.conv = SpectralNorm2d(nn.Conv2d(channels, channels, 1))
        #self.pool = nn.AvgPool2d(2)
    def forward(self, input):
        x = self.block(input)
        skip = x
        #print(x.shape)
        return x+skip

class ResBlock_down2(nn.Module):
    def __init__(self, channels, res_channel, n_blocks):
        super().__init__()
        block = [BlockSN(channels,res_channel) for i in range(n_blocks)]
        self.block = nn.Sequential(*block)
        #self.conv = SpectralNorm2d(nn.Conv2d(channels, channels, 1))
        self.pool = nn.AvgPool2d(2)
    def forward(self, input):
        x = self.block(input)
        skip = input
        #print(x.shape)
        return self.pool(x+skip)

class ResBlock_up(nn.Module):
    def __init__(self, channels, res_channel, n_blocks):
        super().__init__()
        block = [BlockSN(channels,res_channel) for i in range(n_blocks)]
        self.block = nn.Sequential(*block)
        #self.conv = SpectralNorm2d(nn.Conv2d(channels, channels, 1))
        self.up = nn.Upsample(scale_factor=2)
    def forward(self, input):
        input = self.up(input)
        x = self.block(input)
        skip = input
        o = x+skip
        return o

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, scale_factor=2,out_channel = None):
        super().__init__()
        assert scale_factor in [1,2,4,8]
        blocks = [
            nn.Conv2d(in_channel, channel, 3,  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]
        
        blocks.append(ResBlockSN(channel,n_res_channel,n_res_block))

        if scale_factor >= 2:
            blocks.append(ResBlock_down2(channel,n_res_channel,n_res_block))        
        if scale_factor >= 4:
            blocks.append(ResBlock_down2(channel,n_res_channel,n_res_block))
        if scale_factor >= 8:
            blocks.append(ResBlock_down2(channel,n_res_channel,n_res_block))
        if out_channel != None:
            blocks.append(
            nn.Sequential(
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(channel, channel, 3, padding=1)),
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(channel, out_channel, 1)),
            ))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, input):
        return self.blocks(input)

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, scale_factor=2):
        super().__init__()
        assert scale_factor in [2,4]
        blocks = [
            nn.Conv2d(in_channel, channel, 3,  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]
        blocks.append(ResBlockSN(channel,n_res_channel,n_res_block))
        if scale_factor >= 2:
            blocks.append(ResBlock_up(channel,n_res_channel,n_res_block))
        if scale_factor == 4:
            blocks.append(ResBlock_up(channel,n_res_channel,n_res_block))

        blocks.append(
            nn.Sequential(
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(channel, channel, 3, padding=1)),
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(channel, out_channel, 1)),
            ))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.blocks(input)
        return x



class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.array(np.diag(w_u))
        w_u = np.array(np.triu(w_u, 1))
        u_mask = np.array(np.triu(np.ones_like(w_u), 1))
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class BlockCondition(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, latent_channel = 64):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2+latent_channel, in_channel * 4)

        else:
            self.prior = ZeroConv2d(latent_channel, in_channel * 8)

    def forward(self, input, condition=None):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            cond = torch.cat([out,condition],1)
            mean, log_sd = self.prior(cond).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            #zero = torch.zeros_like(out)
            mean, log_sd = self.prior(condition).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None,condition=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                cond = torch.cat([input,condition],1)
                mean, log_sd = self.prior(cond).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                #zero = torch.zeros_like(input)
                cond = condition
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(cond).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed

class Encoder_v2(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, scale_factor=2,out_channel = None):
        super().__init__()
        assert scale_factor in [1,2,4,8]
        blocks = [
            nn.Conv2d(in_channel, channel, 3,  padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]
        
        blocks.append(ResBlockSN(channel,n_res_channel,n_res_block))

        if scale_factor >= 2:
            blocks.append(ResBlock_down2(channel,n_res_channel,n_res_block))        
        if scale_factor >= 4:
            blocks.append(ResBlock_down2(channel,n_res_channel,n_res_block))
        if scale_factor >= 8:
            blocks.append(ResBlock_down2(channel,n_res_channel,n_res_block))
        if out_channel != None:
            blocks.append(
            nn.Sequential(
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(channel, channel, 3, padding=1)),
            nn.ReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(channel, out_channel, 1)),
            ))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        return self.blocks(out)

class GlowCondition(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True, latent_channel = 64,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(BlockCondition(n_channel, n_flow, affine=affine, conv_lu=conv_lu,latent_channel=latent_channel))
            n_channel *= 2
        self.blocks.append(BlockCondition(n_channel, n_flow, split=False, affine=affine,latent_channel=latent_channel))

    def forward(self, input, conditions):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []
        idx = 0

        for block in self.blocks:
            condition = conditions[idx]
            out, det, log_p, z_new = block(out,condition)
            #print(out.shape)
            z_outs.append(z_new)
            logdet = logdet + det
            idx+=1
            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, conditions, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                cond = conditions[-1]
                input = block.reverse(z_list[-1], z_list[-1],cond, reconstruct=reconstruct)
            else:
                cond = conditions[-(i + 1)]
                input = block.reverse(input, z_list[-(i + 1)],cond, reconstruct=reconstruct)

        return input



class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True, conv_lu=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input





class QAE(nn.Module):
    # 32 x 32 x 3 or 64 x 64 x 3
    def __init__(
        self,
        in_channel=3,
        channel=256,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=1,#16, #64
        n_embed=8,#64, #512
        decay=0.99,
        n_flow=32,
        n_block=4,
        affine=False,
        conv_lu = True,
        num_class = 2,
        class_embed_dim = 2,
    ):
        super().__init__()
        config = {      'double_z': False,
      'z_channels': 256,
      'resolution': 256,
      'in_channels': in_channel,
      'out_ch': 3,
      'ch': 128,
      'ch_mult': [ 1,1,2,2,4],
      'num_res_blocks': 2,
      'attn_resolutions': [40],
      'dropout': 0.0}

        self.enc_b = Encoder_base(**config)
        #in_channel*4, channel, n_res_block, n_res_channel, scale_factor=1)
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.class_embed = nn.Embedding(num_class,class_embed_dim)
        self.class_embed_dim = class_embed_dim
        self.post_quant_conv = nn.Conv2d(embed_dim+class_embed_dim, channel, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)

        self.dec = Decoder_base(**config)
        self.input_shape = None

    def forward(self, input, y):
        quant_b, diff_b, id_b = self.encode(input)

        dec = self.decode(quant_b, y)

        return dec, diff_b

    def encode(self, input):
        enc_b = self.enc_b(input) # b * c * 8 * 8 
        #enc_t = self.enc_t(enc_b)
        #print(enc_b.shape)
        #quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        #quant_t, diff_t, id_t = self.quantize_t(quant_t)
        #quant_t = quant_t.permute(0, 3, 1, 2)
        #diff_t = diff_t.unsqueeze(0)

        #dec_t = self.dec_t(quant_t)
        #enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) # b  * 8 * 8 * e
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return  quant_b, diff_b, id_b

    def decode(self, quant_b, y):
        #upsample_t = self.upsample_t(quant_t)
        y_embed = self.class_embed(y).view(-1,self.class_embed_dim,1,1)
        h = quant_b.shape[2]
        w = quant_b.shape[3]

        quant = torch.cat([quant_b,y_embed.repeat(1,1,h,w)],1)
        quant_b = self.post_quant_conv(quant)
        dec = self.dec(quant_b)

        return dec

    def decode_code(self, code_b):
        #print(code_b.shape)
        #quant_t = self.quantize_t.embed_code(code_t)
        #quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_b)

        return dec



class QAE_v2(nn.Module):
    # 32 x 32 x 3 or 64 x 64 x 3
    def __init__(
        self,
        in_channel=3,
        channel=256,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=1,#16, #64
        n_embed=8,#64, #512
        decay=0.99,
        n_flow=32,
        n_block=4,
        affine=False,
        conv_lu = True,
        num_class = 2,
        class_embed_dim = 2,
    ):
        config = {      'double_z': False,
          'z_channels': channel,
          'resolution': 32,
          'in_channels': in_channel,
          'out_ch': 3,
          'ch': 128,
          'ch_mult': [ 4,4], # 32, 16 , 8 
          'num_res_blocks': 2,
          'attn_resolutions': [16],
          'dropout': 0.0}
        super().__init__()
        self.enc_b = Encoder_base(**config)
        #Encoder_v2(in_channel*4, channel, n_res_block, n_res_channel, scale_factor=1)
        #Encoder_base(**config)
        #Encoder_v2(in_channel, channel, n_res_block, n_res_channel, scale_factor=2)
        #E
        #Encoder_v2(in_channel*4, channel, n_res_block, n_res_channel, scale_factor=1)
        self.enc_1 = Encoder(embed_dim, channel, n_res_block, n_res_channel, scale_factor=8,out_channel=channel)
        self.enc_2 = Encoder(embed_dim, channel, n_res_block, n_res_channel, scale_factor=4,out_channel=channel)
        self.enc_3 = Encoder(embed_dim, channel, n_res_block, n_res_channel, scale_factor=2,out_channel=channel)
        self.enc_4 = Encoder(embed_dim, channel, n_res_block, n_res_channel, scale_factor=1,out_channel=channel)
        self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)

        self.dec = GlowCondition(
        in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu,latent_channel=channel
        )
        self.input_shape = None


    def forward(self, input, z_list):
        if self.input_shape is None:
            self.input_shape = input.shape[1:]
        quant_b, diff_b, id_b = self.encode(input)
        condition1 = self.enc_1(quant_b)
        condition2 = self.enc_2(quant_b)
        condition3 = self.enc_3(quant_b)
        condition4 = self.enc_4(quant_b)

        conditions = [condition4,condition3,condition2,condition1]


        rec_img = self.dec.reverse(z_list,conditions)
        diff_t = (input - rec_img).pow(2).mean()

        return diff_b,diff_t


    def encode(self, input):

        enc_b = self.enc_b(input) # b * c * 8 * 8 
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) # b  * 8 * 8 * e
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return  quant_b, diff_b, id_b

    def rec(self, image, z):
        dec = self.decode(image,z)
        return dec


    def decode(self, image, z = None):
        quant_b, diff_b, id_b = self.encode(image)
        condition1 = self.enc_1(quant_b)
        condition2 = self.enc_2(quant_b)
        condition3 = self.enc_3(quant_b)
        condition4 = self.enc_4(quant_b)

        conditions = [condition4,condition3,condition2,condition1]

        if z is None:
            raise

        dec = self.dec.reverse(z,conditions)
        
        return dec

    def decode_code(self, code_b, z=None):
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        condition1 = self.enc_1(quant_b)
        condition2 = self.enc_2(quant_b)
        condition3 = self.enc_3(quant_b)
        condition4 = self.enc_4(quant_b)

        conditions = [condition4,condition3,condition2,condition1]
        #dec = self.decode(quant_b)
        if z is None:
            raise
        dec = self.dec.reverse(z,conditions)
        return dec




if __name__ == "__main__":
    def calc_z_shapes(n_channel, input_size, n_flow, n_block):
        z_shapes = []

        for i in range(n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes
    print('Check')
    img = torch.randn([8,3,32,32])
    label = torch.zeros([8,1]).long()
    z_shapes = calc_z_shapes(3,32,32,4)
    z_sample = []
    for z in z_shapes:
        z_new = torch.randn(8, *z) * 0.7
        z_sample.append(z_new)
    #print(z_shapes)
    #raise
    model = QAE_v2()

    model.train()
    a,b = model(img,z_sample)
    print(a,b)
    print((model.encode(img))[0].shape)
    #print(model.encode(img)[0].shape)

    
