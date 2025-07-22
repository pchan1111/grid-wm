import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from einops import repeat, rearrange, reduce
# from pykeops.torch import LazyTensor


@torch.no_grad()
def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.no_grad()
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

# class HardComparisonSTE(torch.autograd.Function):
def print_gpu_memory_usage(message=""):
    """GPUのメモリ使用状況をMB単位で表示するヘルパー関数"""
    allocated = torch.cuda.memory_allocated() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{message} | 使用中: {allocated:.2f} MB | 最大使用量: {max_allocated:.2f} MB")

class SymLogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        target = symlog(target)
        return 0.5*F.mse_loss(output, target)


class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes-1)

        # use register buffer so that bins move with .cuda() automatically
        self.bins: torch.Tensor
        self.register_buffer(
            'bins', torch.linspace(-20, 20, num_classes), persistent=False)

    def forward(self, output, target):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index-1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1-weight)*F.one_hot(index-1, self.num_classes) + weight*F.one_hot(index, self.num_classes)

        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        return loss.mean()

    def decode(self, output):
        return symexp(F.softmax(output, dim=-1) @ self.bins)

class SeparationLoss(nn.Module):
    def __init__(self, sep_threshold, conf):
        super().__init__()
        self.sep_threshold = sep_threshold
        self.temperature = conf.Models.WorldModel.SeparationLoss.SigmoidTemperature
        self.scaling_factor = 1.0
        self.dist_temperature = conf.Models.WorldModel.SeparationLoss.ExponentialTemperature
        self.att_loss_gate = conf.Models.WorldModel.SeparationLoss.AttractionLossGate
        self.rep_loss_gate = conf.Models.WorldModel.SeparationLoss.RepulsionLossGate

    def forward(self, prior, h):
        B, L, K, C = prior.shape
        eps = 1e-10

        # --- 1. Calculate JSD ---
        p1 = prior.unsqueeze(2) # (B, L, 1, K, C)
        p2 = prior.unsqueeze(1) # (B, 1, L, K, C)
        dist_p1 = Categorical(logits=p1.contiguous())
        dist_p2 = Categorical(logits=p2.contiguous())
        m = 0.5 * (dist_p1.probs + dist_p2.probs) # (B, L, L, K, C)
        dist_m = Categorical(probs=m.contiguous())

        # kl_p_m = (p1 * (torch.log(p1 + eps) - torch.log(m + eps))).sum(dim=-1) # (B, L, L, K)
        # kl_q_m = (p2 * (torch.log(p2 + eps) - torch.log(m + eps))).sum(dim=-1) 
        kl_p_m = torch.distributions.kl.kl_divergence(dist_p1, dist_m) # (B, L, L, K)
        kl_q_m = torch.distributions.kl.kl_divergence(dist_p2, dist_m)

        jsd_matrix= 0.5 * (kl_p_m + kl_q_m)
        jsd_matrix = jsd_matrix.sum(dim=-1) * self.scaling_factor # (B, L, L)

        # --- 2. Calculate two masks ---
        triu_mask = torch.triu(torch.ones(L, L, device=prior.device), diagonal=1)

        # Attraction mask
        soft_att = torch.sigmoid((self.sep_threshold - jsd_matrix) / self.temperature)
        hard_att = (jsd_matrix < self.sep_threshold).float()
        att_mask = (hard_att + soft_att - soft_att.detach()) * triu_mask

        # Repulsion mask
        soft_rep = torch.sigmoid((jsd_matrix - self.sep_threshold) / self.temperature)
        hard_rep = (jsd_matrix >= self.sep_threshold).float()
        rep_mask = (hard_rep + soft_rep - soft_rep.detach()) * triu_mask

        # --- 3. Define loss function ---
        h1 = h.unsqueeze(2) # (B, L, 1, D_h)
        h2 = h.unsqueeze(1) # (B, 1, L, D_h)
        pairwise_mse = ((h1 - h2)**2).mean(dim=-1) # (B, L, L)

        # Attraction loss
        loss_att = (pairwise_mse * att_mask).sum() / (att_mask.sum() + eps)
        loss_att = torch.where(loss_att > self.att_loss_gate, loss_att, 0.0)

        # Repulsion loss
        repulsion_enery = torch.exp(-pairwise_mse / self.dist_temperature)
        loss_rep = (repulsion_enery * rep_mask).sum() / (rep_mask.sum() + eps)
        loss_rep = torch.where(loss_rep > self.rep_loss_gate, loss_rep, 0.0)

        # --- 5. Statistics for debugging ---
        stats = {}
        stats["pairwise_mse_mean"] = pairwise_mse.mean().item()
        stats["pairwise_mse_std"] = pairwise_mse.std().item()
        stats["att_pairs_ratio"] = (att_mask.sum() / (triu_mask.sum() * B)).item()
        stats["rep_pairs_ratio"] = (rep_mask.sum() / (triu_mask.sum() * B)).item()
        stats["loss_att"] = loss_att.item()
        stats["loss_rep"] = loss_rep.item()
        stats["soft_att_mean"] = soft_att.mean().item()
        stats["soft_att_std"] = soft_att.std().item()
        stats["soft_rep_mean"] = soft_rep.mean().item()
        stats["soft_rep_std"] = soft_rep.std().item()
        
        valid_jsd_values = jsd_matrix[triu_mask.unsqueeze(0).expand(B, L, L).bool()]
        stats["mean_jsd"] = valid_jsd_values.mean().item()
        stats["std_jsd"] = valid_jsd_values.std().item()
        # print("loss_att:", stats["loss_att"], "loss_rep:", stats["loss_rep"])
        # print("mean_jsd:", stats["mean_jsd"], "std_jsd:", stats["std_jsd"])
        # print("sep_threshold:", self.sep_threshold)
        
        return  loss_att, loss_rep, stats
    
if __name__ == "__main__":
    loss_func = SymLogTwoHotLoss(255, -20, 20)
    output = torch.randn(1, 1, 255).requires_grad_()
    target = torch.ones(1).reshape(1, 1).float() * 0.1
    print(target)
    loss = loss_func(output, target)
    print(loss)

    # prob = torch.ones(1, 1, 255)*0.5/255
    # prob[0, 0, 128] = 0.5
    # logits = torch.log(prob)
    # print(loss_func.decode(logits), loss_func.bins[128])
