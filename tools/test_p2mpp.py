#!/usr/bin/env python3

import argparse
import json
import sys
import os
from pathlib import Path
import logging
import glob
import tqdm
import numpy as np

import torch

from detectron2.utils.logger import setup_logger
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shapenet.evaluation.eval import compare_points
import shapenet.utils.vis as vis_utils
from ocrtoc.data.mesh_vox_multi_view import MeshVoxMultiViewDataset

logger = logging.getLogger("test_p2mpp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default="./datasets/ocrtoc/3d_dataset",
        help="Path to OCRTOC dataset",
    )
    parser.add_argument(
        "--predict_dir",
        type=Path,
        default="predict",
        help="Path to prediction dir",
    )
    parser.add_argument(
        "--p2mpp_dir",
        type=Path,
        default="../Pixel2MeshPlusPlus",
        help="Path Pixel2MeshPlusPlus repo",
    )
    parser.add_argument(
        "--splits_file",
        type=Path,
        default="./datasets/ocrtoc/ocrtoc_splits_val05.json",
        help="Path to the splits file. This is used for final checking",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def predict_path_to_scene_name(predict_file):
    return "_".join(Path(predict_file).stem.split("-")[0].split("_")[:-1])


def main(args):
    setup_logger(name="test_p2mpp")

    faces = np.loadtxt(args.p2mpp_dir / 'data/face3.obj', dtype='|S32')
    face_indices = torch.from_numpy(
        faces[:, 1:].astype(np.int64) - 1
    ).unsqueeze(0).cuda()

    category_dict, scene_category_dict = MeshVoxMultiViewDataset.get_category_dict(args.dataset_root)

    with open(str(args.splits_file), "r") as f:
        splits = json.load(f)["test"]

    test_scenes = {scene['scene']: scene['image_ids'] for scene in splits}

    xyz_list_path = glob.glob(str(args.predict_dir / "*_predict.xyz"))
    xyz_list_path = [
        i for i in xyz_list_path if predict_path_to_scene_name(i) in test_scenes
    ]

    class_names = {i: i for i in category_dict.keys()}

    num_instances = {i: 0 for i in class_names}
    chamfer = {i: 0 for i in class_names}
    normal = {i: 0 for i in class_names}
    f1_01 = {i: 0 for i in class_names}
    f1_02 = {i: 0 for i in class_names}
    f1_03 = {i: 0 for i in class_names}
    f1_04 = {i: 0 for i in class_names}
    f1_05 = {i: 0 for i in class_names}

    num_batch_evaluated = 0
    for pred_idx, pred_path in tqdm.tqdm(
        enumerate(xyz_list_path), total=len(xyz_list_path)
    ):
        sids = [scene_category_dict[predict_path_to_scene_name(pred_path)]]
        for sid in sids:
            num_instances[sid] += 1

        pred_verts = np.loadtxt(pred_path)
        pred_verts = torch.from_numpy(pred_verts).unsqueeze(0).float().cuda()
        meshes = Meshes(pred_verts, face_indices)

        gt_path = pred_path.replace('_predict', '_ground')
        gt = np.loadtxt(gt_path)
        gt_points = torch.from_numpy(gt[:, :3]).unsqueeze(0).float().cuda()
        gt_normals = torch.from_numpy(gt[:, 3:]).unsqueeze(0).float().cuda()
        batch = {"points": gt_points, "normals": gt_normals}

        cur_metrics = compare_points(batch, meshes)
        cur_metrics["verts_per_mesh"] = meshes.num_verts_per_mesh().cpu()
        cur_metrics["faces_per_mesh"] = meshes.num_faces_per_mesh().cpu()

        for i, sid in enumerate(sids):
            chamfer[sid] += cur_metrics["Chamfer-L2"][i].item()
            normal[sid] += cur_metrics["AbsNormalConsistency"][i].item()
            f1_01[sid] += cur_metrics["F1@%f" % 0.1][i].item()
            f1_02[sid] += cur_metrics["F1@%f" % 0.2][i].item()
            f1_03[sid] += cur_metrics["F1@%f" % 0.3][i].item()
            f1_04[sid] += cur_metrics["F1@%f" % 0.4][i].item()
            f1_05[sid] += cur_metrics["F1@%f" % 0.5][i].item()

        num_batch_evaluated += 1

        # save_obj('/tmp/tmp2.obj', meshes.verts_list()[0], meshes.faces_list()[0])
        # vert = np.hstack((np.full([predict.shape[0], 1], 'v'), predict))
        # mesh = np.vstack((vert, faces))
        # np.savetxt('/tmp/tmp.obj', mesh, fmt='%s', delimiter=' ')

    vis_utils.print_instances_class_histogram(
        num_instances,
        class_names,
        {
            "chamfer": chamfer, "normal": normal,
            "f1_01": f1_01, "f1_02": f1_02, "f1_03": f1_03, "f1_04": f1_04, "f1_05": f1_05
        },
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
