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
import pycolmap
from scipy.spatial.transform import Rotation

import torch

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj, load_ply, save_obj, save_ply
from pytorch3d.transforms import Transform3d

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from shapenet.evaluation.eval import _compute_sampling_metrics
from meshrcnn.utils.metrics import _sample_meshes

from test_dvr_all import get_scene_category_dict

logger = logging.getLogger("test_idr")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_dir", type=Path)
    parser.add_argument("idr_dataset_root", type=Path)
    parser.add_argument("ocrtoc_dataset_root", type=Path)
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


def get_idr_gt_camera_poses(cameras_dict, K):
    # K = np.asarray([
    #     910.6182251, 0, 649.11608887,
    #     0, 910.69207764, 362.55999756,
    #     0, 0, 1
    # ]).reshape((3, 3))

    K_inv = np.linalg.inv(K)
    out = {}

    for key, val in cameras_dict.items():
        if not key.startswith("world_mat_"):
            continue
        Tcw = K_inv @ val[:3, :]
        Tcw = np.concatenate((Tcw, [[0, 0, 0, 1]]), axis=0)
        Twc = np.linalg.inv(Tcw)
        img_id = int(key.split("_")[-1])
        out[img_id] = Twc

    return out


def get_ocrtoc_gt_camera_poses(ocrtoc_dataset_root, scene_name):
    camera_file = glob.glob(
        str(ocrtoc_dataset_root) + f"/scenes/**/{scene_name}/camera_poses.npy",
        recursive=True
    )
    if not camera_file:
        print("no camera poses in ", ocrtoc_dataset_root, scene_name)
        return None, None

    camera_file = camera_file[0]

    poses = np.load(camera_file, allow_pickle=True)[()]
    K = np.load(
        Path(camera_file).parents[0] / 'color_camK.npy', allow_pickle=True
    )[()]
    return poses, K


def get_pose_error(pose1, pose2):
    T_err = np.linalg.inv(pose1) @ pose2
    r_err = Rotation.from_matrix(T_err[:3, :3]).as_euler('xyz', degrees=False)
    r_err = np.linalg.norm(r_err)
    t_err = np.linalg.norm(T_err[:3, -1])
    return r_err + t_err


def find_camera_correspondences(poses1, poses2):
    corres = {}
    for pose1_id, pose1 in poses1.items():
        errs = {
            pose2_id: get_pose_error(pose1, pose2)
            for pose2_id, pose2 in poses2.items()
        }
        corres[pose1_id] = min(errs, key=errs.get)
    return corres


def get_colmap_poses(reconstruction):
    colmap_est = {}
    for image in reconstruction.images.values():
        image_id = int(image.name.split(".")[0])
        # colmap quaternion is [w, x, y, z]
        qcw = image.qvec
        # scipy expects [x, y, z, w]
        Rcw = Rotation.from_quat([qcw[1], qcw[2], qcw[3], qcw[0]]).as_matrix()
        tcw = image.tvec
        Tcw = np.eye(4)
        Tcw[:3, :3] = Rcw
        Tcw[:3, -1] = tcw
        Twc = np.linalg.inv(Tcw)
        colmap_est[image_id] = {
            "pose": Twc,
            "name": image.name
        }
        # print(image_id, image, dir(image))

    return colmap_est


def parse_colmap(pred_dir, idr_dataset_root, gt_poses):
    camera_file = glob.glob(str(pred_dir) + "/**/cameras.bin", recursive=True)
    if not camera_file:
        print("no cameras.bin in", pred_dir)
        return None
    colmap_model_dir = Path(camera_file[0]).parents[0]
    print(Path(camera_file[0]), colmap_model_dir)

    reconstruction = pycolmap.Reconstruction(colmap_model_dir)
    colmap_est = get_colmap_poses(reconstruction)

    image_ids = sorted(list(colmap_est.keys()))
    image_names = [colmap_est[image_id]["name"] for image_id in image_ids]
    locations = [gt_poses[image_id][:3, -1] for image_id in image_ids]
    sim3_transform = reconstruction.align_robust(
        image_names, locations, min_common_images=3,
        max_error=0.05, min_inlier_ratio=0.1
    )

    sim3_transform = np.asarray(sim3_transform.matrix)
    colmap_est = get_colmap_poses(reconstruction)

    align_err = {
        image_id: get_pose_error(
            colmap_est[image_id]["pose"], gt_poses[image_id]
        )
        for image_id in image_ids
    }
    print("align_err:\n", align_err)

    # for camera_id, camera in reconstruction.cameras.items():
    #     print(camera_id, camera)

    pred_meshes = glob.glob(str(pred_dir) + "/**/" + "fused.ply", recursive=True)
    if not pred_meshes:
        print("mesh not found in ", pred_dir)
        return None

    # mesh_pred = load_pred_mesh(pred_meshes[0])
    # verts_pred = mesh_pred.verts_list()[0].float()
    verts_pred, _ = load_ply(pred_meshes[0])

    trans = Transform3d(
        matrix=torch.from_numpy(sim3_transform.T).float()
    )
    verts_pred = trans.transform_points(verts_pred.float())

    return verts_pred.unsqueeze(0)


@torch.no_grad()
def compare_points(batch, pred_points):
    min_gt, _ = batch["points"].min(dim=1)
    max_gt, _ = batch["points"].max(dim=1)
    bbox = torch.stack((min_gt, max_gt), dim=-1) # (N, 3, 2)
    long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
    target = 10.0
    scale = target / long_edge

    return _compute_sampling_metrics(
        pred_points * scale, torch.zeros_like(pred_points),
        batch["points"] * scale, torch.zeros_like(batch["points"]),
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], eps=1e-8
    )


def get_object_name(scene_name, ocrtoc_dataset_root):
    object_files = glob.glob(
        str(ocrtoc_dataset_root) + f"/scenes/**/{scene_name}/object_list.txt", recursive=True
    )
    if not object_files:
        print("object list not found. skipping", pred_mesh)
        return None

    object_file = object_files[0]

    with open(object_file, "r") as f:
        object_name = f.read().strip()

    return object_name


def main(pred_dir, idr_dataset_root, ocrtoc_dataset_root):
    scan_id = pred_dir.name[4:]
    cameras_path = idr_dataset_root / f"scan{scan_id}" / "cameras.npz"
    cameras = np.load(cameras_path)
    scene_name = os.path.realpath(str(idr_dataset_root / f"scan{scan_id}")).split("/")[-1]

    poses_ocrtoc, K = get_ocrtoc_gt_camera_poses(ocrtoc_dataset_root, scene_name)
    if poses_ocrtoc is None or K is None:
        return None

    poses_idr = get_idr_gt_camera_poses(cameras, K)

    corresondences = find_camera_correspondences(poses_idr, poses_ocrtoc)
    poses_ocrtoc = {
        new_id: poses_ocrtoc[old_id] for new_id, old_id in corresondences.items()
    }

    print(
        "correspondences:\n",
        [i[1] for i in sorted(corresondences.items(), key=lambda kv: kv[0])]
    )

    verts_pred = parse_colmap(pred_dir, idr_dataset_root, poses_ocrtoc)
    if verts_pred is None:
        return None

    category_dict, scene_category_dict = get_scene_category_dict(ocrtoc_dataset_root)
    category = scene_category_dict[scene_name]

    print("scan"+scan_id, scene_name, category)

    object_name = get_object_name(scene_name, ocrtoc_dataset_root)
    if object_name is None:
        return

    mesh_gt = ocrtoc_dataset_root / "models" / category / object_name / 'visual.ply'
    mesh_gt = load_gt_mesh(mesh_gt)

    mesh_gt_points, mesh_gt_normals = _sample_meshes(mesh_gt, 10000)
    batch = {"points": mesh_gt_points, "normals": mesh_gt_normals}
    gt_centroid = mesh_gt_points[0].mean(axis=0).float()

    # move to GT centroid
    pred_centroid = verts_pred[0].mean(axis=0).float()
    translation = gt_centroid - pred_centroid

    trans = Transform3d().translate(*(translation.tolist()))
    verts_pred = trans.transform_points(verts_pred.float())

    cur_metrics = compare_points(batch, verts_pred)
    cur_metrics = {
        k: v.item() if torch.is_tensor(v) else v
        for k, v in cur_metrics.items()
    }
    print(json.dumps(cur_metrics, indent=4))
    print("\n\n\n")

    save_ply('/tmp/mesh_pred_scaled.ply', verts_pred[0])
    # save_obj(
    #     '/tmp/mesh_gt.obj',
    #     mesh_gt.verts_list()[0], mesh_gt.faces_list()[0]
    # )

    return {
        "metrics": cur_metrics,
        "scene_name": scene_name,
        "category": category
    }



if __name__ == "__main__":
    args = parse_args()
    main(**args.__dict__)

