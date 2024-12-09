# models/correlation3_unscaled.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.common import ConvBlock, ConvLSTMCell
from utils.losses import L1Truncated


class FPNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, recurrent=False):
        super(FPNEncoder, self).__init__()
        # In a noisy scenario, you may have more or fewer in_channels
        # Adjust accordingly or ensure dataset outputs correct shape.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.recurrent = recurrent

        self.conv_bottom_0 = ConvBlock(in_channels, 32, n_convs=2, kernel_size=1, padding=0, downsample=False)
        self.conv_bottom_1 = ConvBlock(32, 64, n_convs=2, kernel_size=5, padding=0, downsample=False)
        self.conv_bottom_2 = ConvBlock(64, 128, n_convs=2, kernel_size=5, padding=0, downsample=False)
        self.conv_bottom_3 = ConvBlock(128, 256, n_convs=2, kernel_size=3, padding=0, downsample=True)
        self.conv_bottom_4 = ConvBlock(256, out_channels, n_convs=2, kernel_size=3, padding=0, downsample=False)

        if self.recurrent:
            self.conv_rnn = ConvLSTMCell(out_channels, out_channels, 1)

        self.conv_lateral_3 = nn.Conv2d(256, out_channels, 1, bias=True)
        self.conv_lateral_2 = nn.Conv2d(128, out_channels, 1, bias=True)
        self.conv_lateral_1 = nn.Conv2d(64, out_channels, 1, bias=True)
        self.conv_lateral_0 = nn.Conv2d(32, out_channels, 1, bias=True)

        self.conv_dealias_3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv_dealias_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv_dealias_1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv_dealias_0 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.conv_out = nn.Sequential(
            ConvBlock(out_channels, out_channels, n_convs=1, kernel_size=3, padding=1, downsample=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )

        self.conv_bottleneck_out = nn.Sequential(
            ConvBlock(out_channels, out_channels, n_convs=1, kernel_size=3, padding=1, downsample=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
        )

    def reset(self):
        if self.recurrent:
            self.conv_rnn.reset()

    def forward(self, x):
        # Basic input validation
        if x.dim() != 4:
            raise ValueError("Input to FPNEncoder must be a 4D tensor (B,C,H,W).")

        c0 = self.conv_bottom_0(x)
        c1 = self.conv_bottom_1(c0)
        c2 = self.conv_bottom_2(c1)
        c3 = self.conv_bottom_3(c2)
        c4 = self.conv_bottom_4(c3)

        p4 = c4
        p3 = self.conv_dealias_3(self.conv_lateral_3(c3) + F.interpolate(p4, c3.shape[2:], mode='bilinear'))
        p2 = self.conv_dealias_2(self.conv_lateral_2(c2) + F.interpolate(p3, c2.shape[2:], mode='bilinear'))
        p1 = self.conv_dealias_1(self.conv_lateral_1(c1) + F.interpolate(p2, c1.shape[2:], mode='bilinear'))
        p0 = self.conv_dealias_0(self.conv_lateral_0(c0) + F.interpolate(p1, c0.shape[2:], mode='bilinear'))

        if self.recurrent:
            p0 = self.conv_rnn(p0)

        return self.conv_out(p0), self.conv_bottleneck_out(c4)


class JointEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JointEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels, 64, n_convs=2, downsample=True)
        self.conv2 = ConvBlock(64, 128, n_convs=2, downsample=True)
        self.convlstm0 = ConvLSTMCell(128, 128, 3)
        self.conv3 = ConvBlock(128, 256, n_convs=2, downsample=True)
        self.conv4 = ConvBlock(256, 256, kernel_size=3, padding=0, n_convs=1, downsample=False)

        self.flatten = nn.Flatten()
        embed_dim = 256
        num_heads = 8
        self.multihead_attention0 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.prev_x_res = None
        self.gates = nn.Linear(2 * embed_dim, embed_dim)
        self.ls_layer = LayerScale(embed_dim)

        self.fusion_layer0 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1),
        )
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LeakyReLU(0.1)
        )

    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None

    def forward(self, x, attn_mask=None):
        if x.dim() != 4:
            raise ValueError("Input to JointEncoder must be a 4D tensor (B,C,H,W).")

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlstm0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        if self.prev_x_res is None:
            self.prev_x_res = torch.zeros_like(x)

        x = self.fusion_layer0(torch.cat((x, self.prev_x_res), 1))

        x_attn = x[None, :, :].detach()  # (1, B, D)
        if self.training and attn_mask is not None:
            x_attn, _ = self.multihead_attention0(query=x_attn, key=x_attn, value=x_attn, attn_mask=attn_mask.bool())
        else:
            x_attn, _ = self.multihead_attention0(query=x_attn, key=x_attn, value=x_attn)
        x_attn = x_attn.squeeze(0)

        x = x + self.ls_layer(x_attn)
        gate_weight = torch.sigmoid(self.gates(torch.cat((self.prev_x_res, x), 1)))
        x = self.prev_x_res * gate_weight + x * (1 - gate_weight)

        self.prev_x_res = x
        x = self.output_layers(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class TrackerNetC(nn.Module):
    def __init__(
        self,
        representation="time_surfaces_1",
        max_unrolls=16,
        n_vis=8,
        feature_dim=1024,
        patch_size=31,
        init_unrolls=1,
        input_channels=2,  # Adjust based on dataset outputs (target + event channels)
        noise_training=True,
    ):
        super(TrackerNetC, self).__init__()

        self.representation = representation
        self.max_unrolls = max_unrolls
        self.n_vis = n_vis
        self.patch_size = patch_size
        self.init_unrolls = init_unrolls
        self.noise_training = noise_training  # flag if training on noisy data

        # Ensure input_channels aligns with dataset output
        # Suppose half goes to target encoder, half to reference encoder
        self.channels_in_per_patch = input_channels // 2

        self.feature_dim = feature_dim
        self.redir_dim = 128

        self.reference_encoder = FPNEncoder(1, self.feature_dim)  # reference branch
        self.target_encoder = FPNEncoder(self.channels_in_per_patch, self.feature_dim)

        self.reference_redir = nn.Conv2d(self.feature_dim, self.redir_dim, 3, padding=1)
        self.target_redir = nn.Conv2d(self.feature_dim, self.redir_dim, 3, padding=1)
        self.softmax = nn.Softmax(dim=2)

        self.joint_encoder = JointEncoder(
            in_channels=1 + 2 * self.redir_dim,
            out_channels=512
        )
        self.predictor = nn.Linear(512, 2, bias=False)

        self.loss = L1Truncated(patch_size=patch_size)
        self.name = f"corr_{self.representation}"

        self.f_ref, self.d_ref = None, None

    def reset(self):
        self.d_ref, self.f_ref = None, None
        self.joint_encoder.reset()

    def forward(self, x, attn_mask=None):
        if x.dim() != 4:
            raise ValueError("Input to TrackerNetC must be a 4D tensor (B,C,H,W).")

        # x shape: [B, input_channels, H, W]
        # According to original logic:
        # target is x[:, :self.channels_in_per_patch, :, :]
        # reference is x[:, self.channels_in_per_patch:, :, :]

        f0, _ = self.target_encoder(x[:, : self.channels_in_per_patch, :, :])

        if self.f_ref is None:
            self.f_ref, self.d_ref = self.reference_encoder(x[:, self.channels_in_per_patch :, :, :])
            self.f_ref = self.reference_redir(self.f_ref)

        # Correlation
        f_corr = (f0 * self.d_ref).sum(dim=1, keepdim=True)  # B,1,H,W
        f_corr = self.softmax(f_corr.view(-1, 1, self.patch_size * self.patch_size)).view(-1, 1, self.patch_size, self.patch_size)

        # Feature re-direction
        f = torch.cat([f_corr, self.target_redir(f0), self.f_ref], dim=1)
        f = self.joint_encoder(f, attn_mask)
        f = self.predictor(f)

        return f
