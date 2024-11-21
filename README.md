# Tooth Landmark Detection

End-to-End 3D Tooth Landmark Detection

This work is the third solution of MICCAI 2024 Challenge [3DTeethLand](https://www.synapse.org/Synapse:syn57400900/wiki/627259).

It is also an extended application of [KeypointDETR](https://github.com/bibi547/KeypointDETR) for tooth landmark detection.

## Run

### Data preprocessing
1. First, individual teeth are cropped based on the segmentation ground truth provided by [Teeth3DS](https://github.com/abenhamadou/3DTeethSeg22_challenge). [scripts/seg_to_single/segment_patch](scripts/seg_to_single.py)
2. Assign ground truth landmarks to each tooth patch. [scripts/seg_to_single/tooth_landmarks](scripts/seg_to_single.py)
3. Generate the geodesic distance maps for all landmarks on each tooth patch. [scripts/geodesic_distance](scripts/geodesic_distance.py)

### Train
```
python train.py
```

### Test
```
python test.py
```

### Inference
1. First, the 3D tooth segmentation method is executed to crop tooth patches from the segmentation results.
2. Run TL-DETR to detect landmarks on each tooth and map back to the original jaw models.

## Cite
```
@inproceedings{jin2024keypointdetr,
  title={KeypointDETR: an end-to-end 3d keypoint detector},
  author={Jin, Hairong and Shen, Yuefan and Lou, Jianwen and Zhou, Kun and Zheng, Youyi},
  booktitle={ECCV},
  year={2024}
}
```

