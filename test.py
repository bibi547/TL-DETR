import os
import tqdm
import click
import torch
import trimesh

from pl_model import LitModel
from data.st_data import TeethLandDataset

@click.command()
@click.option('--checkpoint', type=str,
              default='E:/code/teeth_landmark_challenge/TeethLand_1_git/runs/tooth_landmark/version_0/checkpoints/best_31.ckpt')  # saved ckpt
@click.option('--gpus', default=1)
def run(checkpoint, gpus):
    model = LitModel.load_from_checkpoint(checkpoint).cuda()
    model.eval()

    args = model.hparams.args
    test_file = 'test.txt'
    dataset = TeethLandDataset(args, test_file, False)

    for i in tqdm.tqdm(range(len(dataset))):
        feats, t_idx, _, _, _ = dataset[i]
        off_file, _, _, _ = dataset.files[i]
        # mesh
        print(off_file)
        mesh = trimesh.load(off_file)
        vs = mesh.vertices
        vs_offset = vs.mean(0)
        feats = feats.unsqueeze(0).cuda()

        # pred
        with torch.no_grad():
            pts, p_labels = model.infer(feats)
            pts = pts + vs_offset

        # continue
        pred_pts = [trimesh.primitives.Sphere(radius=0.2, center=pt).to_mesh() for pt in pts]
        for pt in pred_pts:
            pt.visual.vertex_colors = (0, 255, 0, 255)
        trimesh.Scene([mesh] + pred_pts).show()


if __name__ == "__main__":
    run()
