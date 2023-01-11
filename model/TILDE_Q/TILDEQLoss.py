import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class TILDEQLoss(nn.Module):
    def __init__(self, alpha=0.99, beta=0.5, scale=0.005):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.scale = scale

    def forward(self, y, y_hat):
        as_loss = self.amplitude_shift_loss(y, y_hat)
        ps_loss = self.phase_shift_loss(y, y_hat)
        ua_loss = self.uniform_amplification_loss(y, y_hat)

        loss = self.scale * (self.alpha * as_loss + (1 - self.alpha) * ps_loss) + self.beta * ua_loss
        # print('Loss ', loss, as_loss, ps_loss, ua_loss)
        return loss

    def amplitude_shift_loss(self, y, y_hat):
        B, L, D = y.shape
        dist = F.softmax(y - y_hat, dim=1)
        return L * torch.sum(torch.abs(1 / L - dist)) / (B)

    def phase_shift_loss(self, y, y_hat):
        B, L, D = y.shape
        f = fft.rfft(y, dim=1)
        f_real = f.real
        topk_idx = f_real.topk(int(torch.sqrt(torch.Tensor([L]))), dim=1)[1].long()

        # dominant freq
        y_dominant = y[torch.arange(B)[..., None, None], topk_idx, torch.arange(D)[None, None, ...]]
        y_hat_dominant = y_hat[torch.arange(B)[..., None, None], topk_idx, torch.arange(D)[None, None, ...]]
        loss_dominant = torch.sum(torch.abs(y_dominant - y_hat_dominant)) / B

        # noise freq
        mask = torch.ones_like(y).long()
        mask[:, topk_idx[-1], :] = 0
        y_hat_noise = y_hat[mask]
        lose_noise = torch.sum(torch.abs(y_hat_noise)) / B

        loss = loss_dominant + lose_noise

        return loss

    def uniform_amplification_loss(self, y, y_hat):
        def R(y1, y2):
            """
            Refer to Autoformer
            :param y1:
            :param y2:
            :return:
            """
            B, L, D = y1.shape
            y1_fft = fft.rfft(y1, dim=1)
            y2_fft = fft.rfft(y2, dim=1)
            res = y1 * torch.conj(y2)
            corr = torch.fft.irfft(res, dim=1)
            return corr

        B, L, D = y.shape
        auto_corr = R(y, y)
        cross_corr = R(y, y_hat)

        loss = torch.sum(torch.sqrt(torch.square(auto_corr - cross_corr))) / B
        return loss
