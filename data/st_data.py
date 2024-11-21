import os
import numpy as np
import random
import json
import torch
import trimesh
from torch.utils.data import Dataset
from data.common import calc_features, geodesic_heatmaps, sample


class TeethLandDataset(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        self.args = args
        self.files = []
        self.num_points = args.num_points
        self.augmentation = args.augmentation if train else False

        with open(os.path.join(args.split_root, split_file)) as f:
            for line in f:
                filename = line.strip()
                teeth_root = os.path.join(self.args.patch_root, filename)

                t_files = os.listdir(teeth_root)
                for f in t_files:
                    if f.endswith('.off'):
                        t_idx = os.path.splitext(f)[0]
                        off_file = os.path.join(teeth_root, t_idx + '.off')
                        js_file = os.path.join(teeth_root, t_idx + '.json')
                        gd_file = os.path.join(teeth_root, t_idx + '_f.npy')
                        if os.path.exists(off_file) and os.path.exists(js_file) and os.path.exists(gd_file):
                            self.files.append((off_file, js_file, gd_file, t_idx))
        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        off_file, js_file, gd_file, t_idx = self.files[idx]
        mesh = trimesh.load(off_file)
        landmarks = json.load(open(js_file))
        geo_dists = np.load(gd_file)
        heatmaps = geodesic_heatmaps(geo_dists, self.args)

        vs = mesh.vertices
        vs = vs - vs.mean(0)
        fs, fn = mesh.faces, mesh.face_normals
        fs, fn, heatmaps = sample(fs, fn, heatmaps, self.num_points)
        vs = torch.tensor(vs, dtype=torch.float32)
        fs = torch.tensor(fs, dtype=torch.long)
        fn = torch.tensor(fn, dtype=torch.float32)
        features = calc_features(vs, fs, fn)  # (15, nf)

        cusp_heatmaps = []
        mes_heatmap = np.zeros([1, heatmaps.shape[1]])
        dis_heatmap = np.zeros([1, heatmaps.shape[1]])
        inner_heatmap = np.zeros([1, heatmaps.shape[1]])
        outer_heatmap = np.zeros([1, heatmaps.shape[1]])
        fa_heatmap = np.zeros([1, heatmaps.shape[1]])
        mask = np.zeros([5, self.num_points])
        for i, lm in enumerate(landmarks):
            if lm['class'] == 'Mesial':
                mes_heatmap[0] = heatmaps[i]
                mask[0, :] = 1
            if lm['class'] == 'Distal':
                dis_heatmap[0] = heatmaps[i]
                mask[1, :] = 1
            if lm['class'] == 'InnerPoint':
                inner_heatmap[0] = heatmaps[i]
                mask[2, :] = 1
            if lm['class'] == 'OuterPoint':
                outer_heatmap[0] = heatmaps[i]
                mask[3, :] = 1
            if lm['class'] == 'FacialPoint':
                fa_heatmap[0] = heatmaps[i]
                mask[4, :] = 1
            if lm['class'] == 'Cusp':
                cusp_heatmaps.append(heatmaps[i])
        cusp_heatmaps = np.array(cusp_heatmaps)
        fixed_heatmaps = np.concatenate([mes_heatmap, dis_heatmap, inner_heatmap, outer_heatmap, fa_heatmap], axis=0)

        return features, int(t_idx), \
            torch.tensor(mask, dtype=torch.float32), \
            torch.tensor(fixed_heatmaps, dtype=torch.float32), \
            torch.tensor(cusp_heatmaps, dtype=torch.float32)


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.split_root = 'F:/dataset/3DTeethLand/Split_exp'
            self.patch_root = 'F:/dataset/3DTeethLand/patches'
            self.augmentation = True
            self.normalize_pc = True
            self.landmark_std = 0.5
            self.num_points = 5000

    data = TeethLandDataset(Args(), 'train.txt', True)
    i = 0
    for feats, t_idx, mask, oh,  ch in data:
        print(t_idx)
        print(mask.shape)
        print(feats.shape)
        print(oh.shape)
        print(ch.shape)
        i += 1
    print(i)



