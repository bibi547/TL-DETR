import os
import json
import trimesh
import numpy as np

tar_labels_upper = [11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28]
tar_labels_lower = [31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48]


def trans_labels(labels, category):
    if category == 'upper':
        for i, l in enumerate(tar_labels_upper):
            idx = np.argwhere(labels == l)
            labels[idx] = i + 1
    else:
        for i, l in enumerate(tar_labels_lower):
            idx = np.argwhere(labels == l)
            labels[idx] = i + 1
    return labels


def segment_patch(mesh_file, gt_json, write_path):  # single tooth
    mesh = trimesh.load(mesh_file)
    base_name = os.path.basename(mesh_file)
    mesh_name = os.path.splitext(base_name)[0]
    seg_path = os.path.join(write_path, mesh_name)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    category = mesh_name.split('_')[1]  # upper/lower
    vs = mesh.vertices
    fs = mesh.faces
    with open(gt_json, 'r') as fp:
        json_data = json.load(fp)
    gt_v = json_data['labels']
    gt_v = np.array(gt_v, dtype='int64')
    labels = trans_labels(gt_v, category)
    fv_labels = labels[fs]
    for l in range(1, 17):
        write_file = os.path.join(seg_path, str(l) + '.off')
        v_idx = np.argwhere(labels == l).squeeze()
        if v_idx.shape[0] == 0:continue
        vl = vs[v_idx]

        f_idx = np.where(np.all(fv_labels == l, axis=1))[0]
        fl = fs[f_idx]

        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(v_idx)}
        new_faces = np.array([[index_mapping[vertex] for vertex in face] for face in fl])
        patch_mesh = trimesh.Trimesh(vertices=vl, faces=new_faces)
        patch_mesh.export(write_file)
    print(mesh_name)


def tooth_landmarks(teeth_path, j_file):
    tooth_files = []
    annots = json.load(open(j_file))
    landmarks = annots['objects']
    for file in os.listdir(teeth_path):
        if file.endswith('.off'):
            file_path = os.path.join(teeth_path, file)
            tooth_files.append(file_path)
    t_idx_list = []
    for lm_dict in landmarks:
        cate = lm_dict['class']
        coord = np.array(lm_dict['coord'])
        t_idxs = []
        dists = []
        # dists = np.ones(len(tooth_files))
        for i, f in enumerate(tooth_files):
            t_filename = os.path.basename(f)
            t_idx = os.path.splitext(t_filename)[0]
            mesh = trimesh.load(f)
            vs = mesh.vertices
            ds = np.linalg.norm(vs - coord, axis=1)
            min_dist = np.min(ds)
            dists.append(min_dist)
            t_idxs.append(t_idx)
        min_dist_idx = np.argmin(np.array(dists))
        t_idx = t_idxs[min_dist_idx]
        t_idx_list.append(int(t_idx))
    t_idx_list = np.array(t_idx_list)
    for f in tooth_files:
        t_filename = os.path.basename(f)
        t_idx = os.path.splitext(t_filename)[0]
        lm_idx = np.argwhere(t_idx_list == int(t_idx)).squeeze()
        if lm_idx.size == 0:continue
        elif lm_idx.size == 1: dict_list = [landmarks[lm_idx]]
        else: dict_list = [landmarks[idx] for idx in lm_idx]
        write_file = os.path.join(teeth_path, t_idx + '.json')
        with open(write_file, 'w') as json_file:
            json.dump(dict_list, json_file, indent=len(dict_list))

import glob
if __name__ == "__main__":
    mesh_path = ''  # path to Teeth3DS/data
    json_path = ''  # path to 3DTeethLand/Batch
    split_file = ''  # path to training or testing .txt
    patch_path = ''  # save patches path

    with open(split_file) as f:
        for i, line in enumerate(f):
            # if i < 98:continue
            full_filename = line.strip()
            print(full_filename)
            filename = line.strip().split('_')[0]
            category = line.strip().split('_')[1]
            # root = os.path.join(mesh_path, category, filename)
            # mesh_file = os.path.join(root, f'{line.strip()}.obj')
            # seg_file = os.path.join(root, f'{line.strip()}.json')
            # land_file = os.path.join(self.args.anno_dir, f'{line.strip()}__kps.json')
            teeth_path = os.path.join(patch_path, full_filename)
            json_file = os.path.join(json_path, full_filename + '__kpt.json')
            tooth_landmarks(teeth_path, json_file)



