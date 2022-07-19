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
import open3d as o3d

import torch

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, load_ply, save_obj
from pytorch3d.transforms import Transform3d

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shapenet.evaluation.eval import compare_points
from meshrcnn.utils.metrics import _sample_meshes

logger = logging.getLogger("test_idr")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_gt", type=Path)
    parser.add_argument("mesh_pred", type=Path)
    parser.add_argument("cameras", type=Path)
    return parser.parse_args()


def load_pred_mesh(mesh_path):
    if str(mesh_path).endswith(".ply"):
        verts, faces = load_ply(mesh_path)
    elif str(mesh_path).endswith(".obj"):
        verts, faces, _ = load_obj(mesh_path)
        faces = faces.verts_idx
    else:
        sys.exit("Unrecognized format" % str(mesh_path))

    return Meshes(verts=[verts], faces=[faces])


def load_gt_mesh(mesh_path):
    o3d_mesh = o3d.io.read_triangle_mesh(str(mesh_path), True)
    return Meshes(
        verts=[torch.from_numpy(np.asarray(o3d_mesh.vertices)).float()],
        faces=[torch.from_numpy(np.asarray(o3d_mesh.triangles)).float()]
    )


def main(mesh_gt, mesh_pred, cameras):
    mesh_gt = load_gt_mesh(mesh_gt)
    mesh_pred = load_pred_mesh(mesh_pred)
    cameras = np.load(cameras)
    scale_matrix = cameras['scale_mat_0']

    mesh_gt_points, mesh_gt_normals = _sample_meshes(mesh_gt, 10000)
    batch = {"points": mesh_gt_points, "normals": mesh_gt_normals}
    gt_centroid = mesh_gt_points[0].mean(axis=0).float()

    trans = Transform3d(
        matrix=torch.from_numpy(scale_matrix.T).float()
    )
    verts_pred = trans.transform_points(mesh_pred.verts_list()[0].float())

    # move to GT centroid
    pred_centroid = verts_pred.mean(axis=0).float()
    translation = gt_centroid - pred_centroid

    trans = Transform3d().translate(*(translation.tolist()))
    verts_pred = trans.transform_points(verts_pred.float())

    mesh_pred = Meshes(
        verts=[verts_pred],
        faces=mesh_pred.faces_list()
    )

    cur_metrics = compare_points(batch, mesh_pred)
    cur_metrics = {
        k: v.item() if torch.is_tensor(v) else v
        for k, v in cur_metrics.items()
    }
    print(json.dumps(cur_metrics, indent=4))

    save_obj(
        '/tmp/mesh_pred_scaled.obj',
        mesh_pred.verts_list()[0], mesh_pred.faces_list()[0]
    )
    # save_obj(
    #     '/tmp/mesh_gt.obj',
    #     mesh_gt.verts_list()[0], mesh_gt.faces_list()[0]
    # )

    return cur_metrics


if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)
