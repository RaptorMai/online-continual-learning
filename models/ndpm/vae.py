from abc import ABC, abstractmethod
from itertools import accumulate
import torch
import torch.nn as nn

from utils.utils import maybe_cuda
from .loss import bernoulli_nll, logistic_nll, gaussian_nll, laplace_nll
from .component import ComponentG
from .utils import Lambda
from utils.global_vars import *
from utils.setup_elements import input_size_match

class Vae(ComponentG, ABC):
    def __init__(self, params, experts):
        super().__init__(params, experts)
        x_c, x_h, x_w = input_size_match[params.data]
        bernoulli = MODELS_NDPM_VAE_RECON_LOSS == 'bernoulli'
        if bernoulli:
            self.log_var_param = None
        elif MODELS_NDPM_VAE_LEARN_X_LOG_VAR:
            self.log_var_param = nn.Parameter(
                torch.ones([x_c]) * MODELS_NDPM_VAE_X_LOG_VAR_PARAM,
                requires_grad=True
            )
        else:
            self.log_var_param = (
                    maybe_cuda(torch.ones([x_c])) *
                    MODELS_NDPM_VAE_X_LOG_VAR_PARAM
            )

    def forward(self, x):
        x = maybe_cuda(x)
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var, 1)
        return self.decode(z)

    def nll(self, x, y=None, step=None):
        x = maybe_cuda(x)
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var, MODELS_NDPM_VAE_Z_SAMPLES)
        x_mean = self.decode(z)
        x_mean = x_mean.view(x.size(0), MODELS_NDPM_VAE_Z_SAMPLES, *x.shape[1:])
        x_log_var = (
            None if MODELS_NDPM_VAE_RECON_LOSS == 'bernoulli' else
            self.log_var.view(1, 1, -1, 1, 1)
        )
        loss_recon = self.reconstruction_loss(x, x_mean, x_log_var)
        loss_recon = loss_recon.view(x.size(0), MODELS_NDPM_VAE_Z_SAMPLES, -1)
        loss_recon = loss_recon.sum(2).mean(1)
        loss_kl = self.gaussian_kl(z_mean, z_log_var)
        loss_vae = loss_recon + loss_kl

        return loss_vae

    def sample(self, n=1):
        z = maybe_cuda(torch.randn(n, MODELS_NDPM_VAE_Z_DIM))
        x_mean = self.decode(z)
        return x_mean

    def reconstruction_loss(self, x, x_mean, x_log_var=None):
        loss_type = MODELS_NDPM_VAE_RECON_LOSS
        loss = (
            bernoulli_nll if loss_type == 'bernoulli' else
            gaussian_nll if loss_type == 'gaussian' else
            laplace_nll if loss_type == 'laplace' else
            logistic_nll if loss_type == 'logistic' else None
        )
        if loss is None:
            raise ValueError('Unknown recon_loss type: {}'.format(loss_type))

        if len(x_mean.size()) > len(x.size()):
            x = x.unsqueeze(1)

        return (
            loss(x, x_mean) if x_log_var is None else
            loss(x, x_mean, x_log_var)
        )

    @staticmethod
    def gaussian_kl(q_mean, q_log_var, p_mean=None, p_log_var=None):
        # p defaults to N(0, 1)
        zeros = torch.zeros_like(q_mean)
        p_mean = p_mean if p_mean is not None else zeros
        p_log_var = p_log_var if p_log_var is not None else zeros
        # calcaulate KL(q, p)
        kld = 0.5 * (
            p_log_var - q_log_var +
            (q_log_var.exp() + (q_mean - p_mean) ** 2) / p_log_var.exp() - 1
        )
        kld = kld.sum(1)
        return kld

    @staticmethod
    def reparameterize(z_mean, z_log_var, num_samples=1):
        z_std = (z_log_var * 0.5).exp()
        z_std = z_std.unsqueeze(1).expand(-1, num_samples, -1)
        z_mean = z_mean.unsqueeze(1).expand(-1, num_samples, -1)
        unit_normal = torch.randn_like(z_std)
        z = z_mean + unit_normal * z_std
        z = z.view(-1, z_std.size(2))
        return z

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @property
    def log_var(self):
        return (
            None if self.log_var_param is None else
            self.log_var_param
        )


class SharingVae(Vae, ABC):
    def collect_nll(self, x, y=None, step=None):
        """Collect NLL values

        Returns:
            loss_vae: Tensor of shape [B, 1+K]
        """
        x = maybe_cuda(x)

        # Dummy VAE
        dummy_nll = self.experts[0].g.nll(x, y, step)

        # Encode
        z_means, z_log_vars, features = self.encode(x, collect=True)

        # Decode
        loss_vaes = [dummy_nll]
        vaes = [expert.g for expert in self.experts[1:]] + [self]
        x_logits = []
        for z_mean, z_log_var, vae in zip(z_means, z_log_vars, vaes):
            z = self.reparameterize(z_mean, z_log_var, MODELS_NDPM_VAE_Z_SAMPLES)
            if MODELS_NDPM_VAE_PRECURSOR_CONDITIONED_DECODER:
                x_logit = vae.decode(z, as_logit=True)
                x_logits.append(x_logit)
                continue
            x_mean = vae.decode(z)
            x_mean = x_mean.view(x.size(0), MODELS_NDPM_VAE_Z_SAMPLES,
                                 *x.shape[1:])
            x_log_var = (
                None if MODELS_NDPM_VAE_RECON_LOSS == 'bernoulli' else
                self.log_var.view(1, 1, -1, 1, 1)
            )
            loss_recon = self.reconstruction_loss(x, x_mean, x_log_var)
            loss_recon = loss_recon.view(x.size(0), MODELS_NDPM_VAE_Z_SAMPLES,
                                         -1)
            loss_recon = loss_recon.sum(2).mean(1)
            loss_kl = self.gaussian_kl(z_mean, z_log_var)
            loss_vae = loss_recon + loss_kl

            loss_vaes.append(loss_vae)

        x_logits = list(accumulate(
            x_logits, func=(lambda x, y: x.detach() + y)
        ))
        for x_logit in x_logits:
            x_mean = torch.sigmoid(x_logit)
            x_mean = x_mean.view(x.size(0), MODELS_NDPM_VAE_Z_SAMPLES,
                                 *x.shape[1:])
            x_log_var = (
                None if MODELS_NDPM_VAE_RECON_LOSS == 'bernoulli' else
                self.log_var.view(1, 1, -1, 1, 1)
            )
            loss_recon = self.reconstruction_loss(x, x_mean, x_log_var)
            loss_recon = loss_recon.view(x.size(0), MODELS_NDPM_VAE_Z_SAMPLES,
                                         -1)
            loss_recon = loss_recon.sum(2).mean(1)
            loss_kl = self.gaussian_kl(z_mean, z_log_var)
            loss_vae = loss_recon + loss_kl
            loss_vaes.append(loss_vae)

        return torch.stack(loss_vaes, dim=1)

    @abstractmethod
    def encode(self, x, collect=False):
        pass

    @abstractmethod
    def decode(self, z, as_logit=False):
        """
        Decode do not share parameters
        """
        pass


class CnnSharingVae(SharingVae):
    def __init__(self, params, experts):
        super().__init__(params, experts)
        self.precursors = [expert.g for expert in self.experts[1:]]
        first = len(self.precursors) == 0
        nf_base, nf_ext = MODLES_NDPM_VAE_NF_BASE, MODELS_NDPM_VAE_NF_EXT
        nf = nf_base if first else nf_ext
        nf_cat = nf_base + len(self.precursors) * nf_ext

        h1_dim = 1 * nf
        h2_dim = 2 * nf
        fc_dim = 4 * nf
        h1_cat_dim = 1 * nf_cat
        h2_cat_dim = 2 * nf_cat
        fc_cat_dim = 4 * nf_cat

        x_c, x_h, x_w = input_size_match[params.data]

        self.fc_dim = fc_dim
        feature_volume = ((x_h // 4) * (x_w // 4) *
                          h2_cat_dim)

        self.enc1 = nn.Sequential(
            nn.Conv2d(x_c, h1_dim, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(h1_cat_dim, h2_dim, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Lambda(lambda x: x.view(x.size(0), -1))
        )
        self.enc3 = nn.Sequential(
            nn.Linear(feature_volume, fc_dim),
            nn.ReLU()
        )
        self.enc_z_mean = nn.Linear(fc_cat_dim, MODELS_NDPM_VAE_Z_DIM)
        self.enc_z_log_var = nn.Linear(fc_cat_dim, MODELS_NDPM_VAE_Z_DIM)

        self.dec_z = nn.Sequential(
            nn.Linear(MODELS_NDPM_VAE_Z_DIM, 4 * nf_base),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Linear(
                4 * nf_base,
                (x_h // 4) * (x_w // 4) * 2 * nf_base),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            Lambda(lambda x: x.view(
                x.size(0), 2 * nf_base,
                x_h // 4, x_w // 4)),
            nn.ConvTranspose2d(2 * nf_base, 1 * nf_base,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.ConvTranspose2d(1 * nf_base, x_c,
                                       kernel_size=4, stride=2, padding=1)

        self.setup_optimizer()

    def encode(self, x, collect=False):
        # When first component
        if len(self.precursors) == 0:
            h1 = self.enc1(x)
            h2 = self.enc2(h1)
            h3 = self.enc3(h2)
            z_mean = self.enc_z_mean(h3)
            z_log_var = self.enc_z_log_var(h3)

            if collect:
                return [z_mean], [z_log_var], \
                       [h1.detach(), h2.detach(), h3.detach()]
            else:
                return z_mean, z_log_var

        # Second or later component
        z_means, z_log_vars, features = \
            self.precursors[-1].encode(x, collect=True)

        h1 = self.enc1(x)
        h1_cat = torch.cat([features[0], h1], dim=1)
        h2 = self.enc2(h1_cat)
        h2_cat = torch.cat([features[1], h2], dim=1)
        h3 = self.enc3(h2_cat)
        h3_cat = torch.cat([features[2], h3], dim=1)
        z_mean = self.enc_z_mean(h3_cat)
        z_log_var = self.enc_z_log_var(h3_cat)

        if collect:
            z_means.append(z_mean)
            z_log_vars.append(z_log_var)
            features = [h1_cat.detach(), h2_cat.detach(), h3_cat.detach()]
            return z_means, z_log_vars, features
        else:
            return z_mean, z_log_var

    def decode(self, z, as_logit=False):
        h3 = self.dec_z(z)
        h2 = self.dec3(h3)
        h1 = self.dec2(h2)
        x_logit = self.dec1(h1)
        return x_logit if as_logit else torch.sigmoid(x_logit)
