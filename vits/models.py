
import torch

from torch import nn
from torch.nn import functional as F
from vits import attentions
from vits import commons
from vits import modules
from vits import spectrogram
from vits.utils import f0_to_coarse
from vits_decoder.generator import Generator
from vits.modules_grl import SpeakerClassifier

import math

def gaussian_kernel1d(size: int, sigma: float) -> torch.Tensor:
    """Creates a 1D Gaussian kernel."""
    x = torch.arange(-size, size + 1, dtype=torch.float32).cuda(1)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()  # Normalize the kernel

def apply_gaussian_filter1d_batch(batch: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel1d(kernel_size, sigma).view(1, 1, -1)  # Shape (1, 1, kernel_size)
    
    batch = batch.swapaxes(0, 1).unsqueeze(1)

    smoothed = torch.nn.functional.conv1d(batch, kernel, padding=kernel_size, groups=1)

    return smoothed.squeeze(1).swapaxes(0, 1)

def pca(z, extent, gaussian=False):
    time_size = z.shape[-1]
    batch_size = z.shape[0]
    channel_size = z.shape[1]
    orig_z = z
    # brief_z = z.mean(-1)
    # print(z.isnan().any())
    centre_z = z - torch.mean(z.mean(-1), dim=-1).reshape(batch_size, 1, 1)
    z = centre_z / (torch.norm(centre_z, p=2, dim=-1) + 1e-6).unsqueeze(-1)
    # print(z[0, :4, :4])
    importance = torch.bmm(z, z.transpose(1, -1))
    importance = (importance - torch.min(importance)) / (torch.max(importance) - torch.min(importance)+1e-6)
    importance += torch.diag(torch.ones(channel_size)).unsqueeze(0).cuda(1)
    # print(importance[0, :4, :4])
    u, s, vh = torch.linalg.svd(importance)
    s_cp = s.clone()
    threshold = int(extent/100 * 120)
    s_cp[:, -threshold:] = 0.0 # 192
    diag = torch.stack([torch.diag(per_s) for per_s in s_cp])
    norm = torch.linalg.inv(importance)

    importance = torch.bmm(u, diag) # reduced importance
    importance = torch.bmm(importance, vh)    
    normed_importance = torch.bmm(importance, norm)
    # print(normed_importance.shape)
    # print((normed_importance - torch.diag(torch.ones(192).cuda()).unsqueeze(0)).sum())
    # exit()
    # if gaussian == True:
    #     sigma = extent/100 * 5#0.75
    #     orig_z[0, :, :] = apply_gaussian_filter1d_batch(orig_z[0, :, :], 50, sigma=sigma)
    reduced_z = torch.bmm(normed_importance, orig_z)
    # print((reduced_z - orig_z).sum())
    return reduced_z
        
class TextEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 vec_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2)
        self.hub = nn.Conv1d(vec_channels, hidden_channels, kernel_size=5, padding=2)
        self.pit = nn.Embedding(256, hidden_channels)
        self.enc = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, v, z_q, f0):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        v = torch.transpose(v, 1, -1)  # [b, h, t]
        v = self.hub(v) * x_mask
        x = x + v + self.pit(f0).transpose(1, 2)
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        # combine reduced z_q
        min_len = min(z_q.shape[2], m.shape[2])
        z_q = z_q[:, :, :min_len]
        m = m[:, :, :min_len]
        logs = logs[:, :, :min_len]
        x_mask = x_mask[:, :, :min_len]
        z_q1 = (z_q - z_q.mean())/z_q.std()*m.std() + m.mean()
        m = 0.5 * (z_q1 + m)
        z_q2 = (z_q - z_q.mean())/z_q.std()*logs.std() + logs.mean()
        logs = 0.5 * (z_q2 + logs)
        x = m + torch.randn_like(m) * torch.exp(logs)
        z = x * x_mask
        return z, m, logs, x_mask, x

    def baseline(self, x, x_lengths, v, f0):
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        v = torch.transpose(v, 1, -1)  # [b, h, t]
        v = self.hub(v) * x_mask
        x = x + v + self.pit(f0).transpose(1, 2)
        x = self.enc(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        x = m + torch.randn_like(m) * torch.exp(logs)
        z = x * x_mask
        return z, m, logs, x_mask, x

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                x, log_det = flow(x, x_mask, g=g, reverse=reverse)
                total_logdet += log_det
            return x, total_logdet

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, get_x=False):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        if get_x == True:
            return x
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class SynthesizerTrn(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.emb_g = nn.Linear(hp.vits.spk_dim, hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
            hp.vits.vec_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        self.speaker_classifier = nn.Linear(192, 80)
        self.spk_enc = PosteriorEncoder(
            spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)
        self.hp = hp.data

    def forward(self, ppg, vec, pit, spec, spk, ppg_l, spec_l, stft):
        ppg = ppg + torch.randn_like(ppg) * 1  # Perturbation
        vec = vec + torch.randn_like(vec) * 2  # Perturbation
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        
        z_q, m_q, logs_q, spec_mask = self.enc_q(spec, spec_l, g=g)
        # z_q = pca(z_q, extent=100) ##
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, z_q, f0=f0_to_coarse(pit))
        z_slice, pit_slice, ids_slice = commons.rand_slice_segments_with_pitch(
            z_p, pit, spec_l, self.segment_size)
        audio = self.dec(spk, z_slice, pit_slice)

        # SNAC to flow
        z_f, logdet_f = 0, 0#self.flow(z_q, spec_mask, g=spk)
        z_r, logdet_r = 0, 0#self.flow(z_p, spec_mask, g=spk, reverse=True)
        # speaker
        # audio to spec to embeddings (via posterior encoder .enc)
        new_spec = stft.linear_spectrogram(audio.squeeze(1))
        new_spec = new_spec.cuda(1)
        spk_emb = self.spk_enc(new_spec, torch.tensor(new_spec.shape[-1], dtype=torch.long).unsqueeze(0).cuda(1), g=g, get_x=True)
        spk_preds = self.speaker_classifier(spk_emb.mean(-1))
        return audio, ids_slice, spec_mask, (z_f, z_r, z_p, m_p, logs_p, z_q, m_q, logs_q, logdet_f, logdet_r), spk_preds

    def infer(self, ppg, vec, pit, spk, ppg_l):
        ppg = ppg + torch.randn_like(ppg) * 0.0001  # Perturbation
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec(spk, z * ppg_mask, f0=pit)
        return o


class SynthesizerInfer(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        hp
    ):
        super().__init__()
        self.segment_size = segment_size
        self.emb_g = nn.Linear(hp.vits.spk_dim, hp.vits.gin_channels)
        self.enc_p = TextEncoder(
            hp.vits.ppg_dim,
            hp.vits.vec_dim,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            hp.vits.filter_channels,
            2,
            6,
            3,
            0.1,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            16,
            gin_channels=hp.vits.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            hp.vits.inter_channels,
            hp.vits.hidden_channels,
            5,
            1,
            4,
            gin_channels=hp.vits.spk_dim
        )
        self.dec = Generator(hp=hp)

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()
        self.dec.remove_weight_norm()

    def pitch2source(self, f0):
        return self.dec.pitch2source(f0)

    def source2wav(self, source):
        return self.dec.source2wav(source)

    def inference(self, ppg, vec, pit, spk, ppg_l, source, spec, extent, gaussian=False):
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_q, _, _, _ = self.enc_q(spec, \
            torch.tensor(spec.shape[-1]).reshape(1).long().cuda(1), g=g) ## get spec, spec_l, g
        # z_q = pca(z_q, extent, gaussian=gaussian)
        z, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, z_q, f0=f0_to_coarse(pit))
        z, _ = self.flow(z, ppg_mask, g=spk, reverse=True)
        # z = pca(z, extent, gaussian=gaussian)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o
    
    def no_pca_inference(self, ppg, vec, pit, spk, ppg_l, source, spec):
        g = self.emb_g(F.normalize(spk)).unsqueeze(-1)
        z_q, _, _, _ = self.enc_q(spec, \
            torch.tensor(spec.shape[-1]).reshape(1).long().cuda(1), g=g) ## get spec, spec_l, g
        # reduced_z_q = pca(z_q)
        z, m_p, logs_p, ppg_mask, x = self.enc_p(
            ppg, ppg_l, vec, z_q, f0=f0_to_coarse(pit))
        # z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o
    
    def baseline(self, ppg, vec, pit, spk, ppg_l, source, spec=None):
        z_p, m_p, logs_p, ppg_mask, x = self.enc_p.baseline(
            ppg, ppg_l, vec, f0=f0_to_coarse(pit))
        z, _ = self.flow(z_p, ppg_mask, g=spk, reverse=True)
        o = self.dec.inference(spk, z * ppg_mask, source)
        return o
