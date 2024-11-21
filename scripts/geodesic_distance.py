import os
import trimesh
import numpy as np
import json
from sklearn.manifold import Isomap


def geo_distance_metrix(points):
    isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    data_2d = isomap.fit_transform(X=points)
    geo_distance_metrix = isomap.dist_matrix_  # 测地距离矩阵，shape=[n_sample,n_sample]
    return geo_distance_metrix


def write_landmarks_geo_distance_metrix(off_file, json_file, out_file):
    mesh = trimesh.load(off_file)
    # vs = mesh.vertices
    cs = mesh.triangles_center
    landmarks = json.load(open(json_file))
    coords = [lm['coord'] for lm in landmarks]
    coords = np.array(coords)
    if coords.size == 0: return
    distance_metrix = geo_distance_metrix(cs)
    closest_idx = np.argmin(np.linalg.norm(coords[:, np.newaxis, :] - cs[np.newaxis, :, :], axis=2), axis=1)
    geo_dists = distance_metrix[closest_idx]
    np.save(out_file, geo_dists)


if __name__ == '__main__':
    jaw_root = ''  # patches root

    jaw_names = os.listdir(jaw_root)
    for f in jaw_names:
        teeth_path = os.path.join(jaw_root, f)
        print(f)
        for file in os.listdir(teeth_path):
            if file.endswith('.off'):
                basename = os.path.basename(file)
                t_idx = os.path.splitext(basename)[0]
                print(t_idx)
                off_file = os.path.join(teeth_path, t_idx + '.off')
                json_file = os.path.join(teeth_path, t_idx + '.json')
                write_file = os.path.join(teeth_path, t_idx + '_f.npy')
                if os.path.exists(off_file) and os.path.exists(json_file):
                    write_landmarks_geo_distance_metrix(off_file, json_file, write_file)


