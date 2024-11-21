import torch
import numpy as np


def calc_features(vs, fs, fn):
    nf = fs.shape[0]
    fea = torch.empty((15, nf), dtype=torch.float32).to(vs.device)

    vs_in_ts = vs[fs]

    fea[:3, :] = vs_in_ts.mean(1).T[None]  # centers , 3
    fea[3:6, :] = fn.T[None]  # normal, 3
    fea[6:15, :] = (vs_in_ts - vs_in_ts.mean(1, keepdim=True)).reshape((nf, -1)).T[None]
    return fea


def geodesic_heatmaps(geodesic_matrix, args):
    ys = []
    for i, dist in enumerate(geodesic_matrix):
        y = np.exp(- dist ** 2 / (2 * args.landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys).T


def sample(fs, fn, hs, num_points):  # 15xN MxN
    # hs = hs.T
    sample_fs = np.zeros((num_points, fs.shape[1]), dtype=int)
    sample_fn = np.zeros((num_points, fn.shape[1]), dtype=float)
    sample_hs = np.zeros((num_points, hs.shape[1]), dtype=float)

    if len(fs) < num_points:
        n = len(fs)
        sample_fs[:n] = fs
        sample_fn[:n] = fn
        sample_hs[:n] = hs
        idx = np.random.choice(len(fs), num_points - n, replace=True)
        sample_fs[n:] = fs[idx]
        sample_fn[n:] = fn[idx]
        sample_hs[n:] = hs[idx]
    else:
        idx = np.random.permutation(len(fs))[:num_points]
        sample_fs[:] = fs[idx]
        sample_fn[:] = fn[idx]
        sample_hs[:] = hs[idx]

    return sample_fs, sample_fn, sample_hs.T


