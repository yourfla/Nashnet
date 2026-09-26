import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from torch.nn.parameter import Parameter

class SoftAtt(nn.Module):

    def __init__(self):
        super(SoftAtt, self).__init__()
        self.evidence_scale = nn.Parameter(torch.tensor(1.0))
        self.dirichlet_strength = nn.Parameter(torch.tensor(0.5))
        self.conv_block = nn.Sequential(nn.ConvTranspose2d(256, 64, 2, stride=2), nn.BatchNorm2d(64), nn.PReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.pred = nn.Conv2d(64 * 4, 2, kernel_size=3, padding=1)
        self.Soft = Soft()
        self.evidence_conv = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.nash_prior_fusion = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=32, kernel_size=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(in_channels=16, out_channels=6, kernel_size=1), nn.Sigmoid())
        self.nash_spatial_att = nn.Sequential(nn.Conv2d(6, 1, kernel_size=1), nn.Sigmoid())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def compute_evidence(self, x):
        x = torch.clamp(x, -10.0, 10.0)
        return torch.clamp(F.softplus(x) * self.evidence_scale, 0.0, 50.0)

    def apply_nash_prior(self, six_rater_outputs, nash_weights):
        if nash_weights is None:
            return six_rater_outputs
        batch_size = six_rater_outputs[0].shape[0]
        H, W = (six_rater_outputs[0].shape[2], six_rater_outputs[0].shape[3])
        nash_weights_spatial = nash_weights.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 6, H, W)
        nash_prior_map = self.nash_prior_fusion(nash_weights_spatial)
        spatial_attention = self.nash_spatial_att(nash_prior_map)
        modulated_outputs = []
        for i, expert_output in enumerate(six_rater_outputs):
            expert_weight = nash_prior_map[:, i:i + 1, :, :]
            combined_weight = expert_weight * spatial_attention
            modulated_expert = expert_output * (1.0 + combined_weight)
            modulated_outputs.append(modulated_expert)
        return modulated_outputs

    def forward(self, Out_six_rater, fea, out, nash_weights=None):
        Out_six_rater = self.apply_nash_prior(Out_six_rater, nash_weights)
        fea = self.conv_block(fea)
        pred_cup_list = []
        pred_disc_list = []
        evidence_cup_list = []
        evidence_disc_list = []
        for i in range(len(Out_six_rater)):
            pred_disc = torch.sigmoid(Out_six_rater[i][:, 0, :, :]).unsqueeze(1)
            pred_cup = torch.sigmoid(Out_six_rater[i][:, 1, :, :]).unsqueeze(1)
            evidence_disc = self.compute_evidence(Out_six_rater[i][:, 0, :, :]).unsqueeze(1)
            evidence_cup = self.compute_evidence(Out_six_rater[i][:, 1, :, :]).unsqueeze(1)
            pred_disc_list.append(pred_disc)
            pred_cup_list.append(pred_cup)
            evidence_disc_list.append(evidence_disc)
            evidence_cup_list.append(evidence_cup)
        pred_cup = torch.cat(pred_cup_list, dim=1)
        pred_disc = torch.cat(pred_disc_list, dim=1)
        evidence_cup = torch.cat(evidence_cup_list, dim=1)
        evidence_disc = torch.cat(evidence_disc_list, dim=1)
        epsilon = 1e-06
        total_evidence_cup = torch.sum(evidence_cup, dim=1, keepdim=True)
        total_evidence_disc = torch.sum(evidence_disc, dim=1, keepdim=True)
        total_evidence_cup = torch.clamp(total_evidence_cup, epsilon, 100.0)
        total_evidence_disc = torch.clamp(total_evidence_disc, epsilon, 100.0)
        alpha_cup = evidence_cup + self.dirichlet_strength
        alpha_disc = evidence_disc + self.dirichlet_strength
        alpha_0_cup = torch.sum(alpha_cup, dim=1, keepdim=True)
        alpha_0_disc = torch.sum(alpha_disc, dim=1, keepdim=True)
        uncertainty_cup_orig = 1.0 / (total_evidence_cup + 1.0)
        uncertainty_disc_orig = 1.0 / (total_evidence_disc + 1.0)
        uncertainty_cup_dir = 2.0 / (alpha_0_cup + 1.0)
        uncertainty_disc_dir = 2.0 / (alpha_0_disc + 1.0)
        mix_ratio = torch.sigmoid(self.dirichlet_strength)
        uncertainty_cup = uncertainty_cup_orig * (1 - mix_ratio) + uncertainty_cup_dir * mix_ratio
        uncertainty_disc = uncertainty_disc_orig * (1 - mix_ratio) + uncertainty_disc_dir * mix_ratio
        if nash_weights is not None:
            nash_entropy = -(nash_weights * torch.log(nash_weights + epsilon)).sum(dim=1)
            max_entropy = torch.log(torch.tensor(6.0, device=nash_weights.device))
            nash_uncertainty = nash_entropy / max_entropy
            nash_uncertainty_spatial = nash_uncertainty.view(-1, 1, 1, 1).expand_as(uncertainty_cup)
            adaptive_weight = torch.sigmoid(5.0 * (nash_uncertainty_spatial - 0.5))
            uncertainty_cup = adaptive_weight * uncertainty_cup + (1 - adaptive_weight) * nash_uncertainty_spatial
            uncertainty_disc = adaptive_weight * uncertainty_disc + (1 - adaptive_weight) * nash_uncertainty_spatial
        weighted_cup = torch.sum(pred_cup * evidence_cup, dim=1, keepdim=True) / (total_evidence_cup + epsilon)
        weighted_disc = torch.sum(pred_disc * evidence_disc, dim=1, keepdim=True) / (total_evidence_disc + epsilon)
        cup_weight_factor = 1.5
        weighted_cup = weighted_cup * cup_weight_factor
        weighted_cup = torch.clamp(weighted_cup, 0.0, 1.0)
        Att_disc = torch.sigmoid(out[:, 0, :, :]).unsqueeze(1)
        Att_cup = torch.sigmoid(out[:, 1, :, :]).unsqueeze(1)
        soft_u_cup_fea = self.Soft(uncertainty_cup, fea) * 1.2 + fea
        soft_u_disc_fea = self.Soft(uncertainty_disc, fea) * 0.9 + fea
        soft_cup_fea = self.Soft(weighted_cup, fea) * 1.3 + fea
        soft_disc_fea = self.Soft(weighted_disc, fea) + fea
        fea_enhanced = torch.cat([soft_cup_fea, soft_disc_fea, soft_u_cup_fea, soft_u_disc_fea], dim=1)
        out = self.pred(fea_enhanced)
        evidence_outputs = self.evidence_conv(fea)
        return (out, [Att_disc, Att_cup, uncertainty_cup, uncertainty_disc, weighted_cup, weighted_disc, evidence_outputs])

def gkern(kernlen=16, nsig=3):
    interval = (2 * nsig + 1.0) / kernlen
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-08)

class Soft(nn.Module):

    def __init__(self):
        super(Soft, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x
