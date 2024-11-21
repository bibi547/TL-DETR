import numpy as np
import torch
import torch.nn as nn
from .dgcnn import Backbone, SharedMLP1d, EdgeConv, knn


class QueryDecoder(nn.Module):
    def __init__(self, args):
        super(QueryDecoder, self).__init__()

        self.k = args.k

        self.smlp1 = SharedMLP1d([args.emb_dims+args.n_edgeconvs_backbone*64, 256, 64], args.norm)
        self.conv = EdgeConv([64*2, 128], args.k, args.norm)

    def forward(self, x, p):
        x = self.smlp1(x)
        x = self.conv(x, knn(p, self.k))

        return x


class TeethDETR(nn.Module):
    def __init__(self, args):
        super(TeethDETR, self).__init__()

        self.query_num = args.query_num
        self.dynamic = args.dynamic
        self.backbone = Backbone(args)

        self.query_decoder = QueryDecoder(args)

        self.fheats_out = nn.Sequential(SharedMLP1d([128, 64], args.norm),
                                        nn.Dropout(args.dropout),
                                        nn.Conv1d(64, 5, kernel_size=1))

        self.cusp_decoder = nn.Sequential(SharedMLP1d([128, 64], args.norm),
                                          nn.Dropout(args.dropout),
                                          nn.Conv1d(64, args.query_num, kernel_size=1))
        self.cprob_out = nn.Sequential(SharedMLP1d([args.num_points, 1024], args.norm),
                                       SharedMLP1d([1024, 256], args.norm),
                                       SharedMLP1d([256, 64], args.norm),
                                       nn.Dropout(args.dropout),
                                       nn.Conv1d(64, 2, kernel_size=1), )  # 2 x 50

        self.cheats_out = nn.Sequential(SharedMLP1d([args.query_num, args.query_num], args.norm),
                                        nn.Dropout(args.dropout),
                                        nn.Conv1d(args.query_num, args.query_num, kernel_size=1))

    def forward(self, x):

        xyz = x[:, :3, :].contiguous()  # B,3,N

        feat = self.backbone(x)
        feat = self.query_decoder(feat, feat if self.dynamic else xyz)  # B,128,N

        fheats = self.fheats_out(feat)

        cheat_feat = self.cusp_decoder(feat)  # B,M,N
        cprobs = self.cprob_out(cheat_feat.permute(0, 2, 1))  # B,2,M
        cheats = self.cheats_out(cheat_feat)  # B,N,M

        return fheats, cprobs.permute(0, 2, 1), cheats

    def inference(self, x):
        xyz = x[:, :3, :].contiguous()  # B,3,N
        xyz_t = xyz.permute(0,2,1)

        feat = self.backbone(x)
        feat = self.query_decoder(feat, feat if self.dynamic else xyz)  # B,128,N

        fheats = self.fheats_out(feat)

        cheat_feat = self.cusp_decoder(feat)  # B,M,N
        cprobs = self.cprob_out(cheat_feat.permute(0, 2, 1))  # B,2,M
        cheats = self.cheats_out(cheat_feat)  # B,N,M

        xyz_np = xyz_t.squeeze().detach().cpu().numpy()
        fheats_np = fheats.squeeze().detach().cpu().numpy()
        f_idx = np.argmax(fheats_np, axis=1)
        f_xyz = xyz_np[f_idx]
        f_label = np.array([1, 2, 3, 4, 5])

        cprobs = cprobs.squeeze().detach().cpu().numpy()
        cprobs = np.argmax(cprobs, axis=0)
        cheats_np = cheats.squeeze().detach().cpu().numpy()
        c_idx = np.argmax(cheats_np, axis=1)
        c_xyz = xyz_np[c_idx]
        s_idx = np.argwhere(cprobs == 1).squeeze()
        c_xyz = c_xyz[s_idx]
        c_labels = np.ones(c_xyz.shape[0]) * 6

        p_xyz = np.concatenate([f_xyz, c_xyz], axis=0)
        p_label = np.concatenate([f_label, c_labels], axis=0)

        return p_xyz, p_label
