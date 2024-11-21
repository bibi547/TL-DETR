import torch
import torch.nn as nn

from typing import Optional


def knn(x, k: int):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k: int, idx: Optional[torch.Tensor] = None):
    batch_size = x.size(0)
    n_channels = x.size(1)      # fixed
    num_points = x.size(2)      # dynamic
    x = x.view(batch_size, n_channels, -1)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.long()

    idx = idx + idx_base

    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(-1, n_channels)[idx, :]
    feature = feature.view(batch_size, -1, k, n_channels)
    x = x.view(batch_size, -1, 1, n_channels).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def get_norm_layer_1d(norm):
    if norm == 'instance':
        norm_layer = nn.InstanceNorm1d
    elif norm == 'batch':
        norm_layer = nn.BatchNorm1d
    else:
        assert 0, "not implemented"
    return norm_layer


def get_norm_layer_2d(norm):
    if norm == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm == 'batch':
        norm_layer = nn.BatchNorm2d
    else:
        assert 0, "not implemented"
    return norm_layer


class SharedMLP2d(nn.Module):
    def __init__(self, channels, norm):
        super(SharedMLP2d, self).__init__()

        norm_layer = get_norm_layer_2d(norm)

        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(channels[i-1], channels[i], kernel_size=1, bias=False),
                          norm_layer(channels[i]),
                          nn.LeakyReLU(0.2))
            for i in range(1, len(channels))
        ])

    def forward(self, x):
        return self.conv(x)


class SharedMLP1d(nn.Module):
    def __init__(self, channels, norm):
        super(SharedMLP1d, self).__init__()

        norm_layer = get_norm_layer_1d(norm)

        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=False),
                          norm_layer(channels[i]),
                          nn.LeakyReLU(0.2))
            for i in range(1, len(channels))
        ])

    def forward(self, x):
        return self.conv(x)


class EdgeConv(nn.Module):
    def __init__(self, channels, k, norm):
        super(EdgeConv, self).__init__()

        self.k = k
        self.smlp = SharedMLP2d(channels, norm)

    def forward(self, x, idx: Optional[torch.Tensor] = None):
        x = get_graph_feature(x, self.k, idx)       # (N, channels[0], num_points, k)
        x = self.smlp(x)
        x = x.max(dim=-1, keepdim=False)[0]
        return x


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        self.k = args.k
        self.dynamic = args.dynamic

        channel = args.input_channels
        self.convs = nn.ModuleList()
        for _ in range(args.n_edgeconvs_backbone):
            self.convs.append(EdgeConv([channel*2, 64, 64], args.k, args.norm))
            channel = 64

        self.smlp = SharedMLP1d([args.n_edgeconvs_backbone*64, args.emb_dims], args.norm)

        # global pooling
        if args.global_pool_backbone == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif args.global_pool_backbone == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            assert 0

    def forward(self, x):
        idx = knn(x[:, :3, :].contiguous(), self.k)

        xs = []
        for conv in self.convs:
            x = conv(x, None if self.dynamic else idx)
            xs.append(x)
        x = torch.cat(xs, dim=1)

        x = self.smlp(x)
        x_pool = self.pool(x)
        x_pool = x_pool.expand(x_pool.shape[0], x_pool.shape[1], x.shape[2])

        x = torch.cat((x_pool, torch.cat(xs, dim=1)), dim=1)

        return x

