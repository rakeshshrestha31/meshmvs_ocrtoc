import argparse
import json
import logging
import os
import sys
import shutil
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from detectron2.utils.logger import setup_logger
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.structures import Meshes

from shapenet.utils.binvox_torch import read_binvox_coords
from shapenet.utils.coords import (
    SHAPENET_MAX_ZMAX,
    SHAPENET_MIN_ZMIN,
    project_verts, transform_verts,
    voxel_grid_coords, voxel_coords_to_world
)
from shapenet.modeling.voxel_ops import logit

from ocrtoc.data.mesh_vox import MeshVoxDataset

from tools.preprocess_shapenet import align_bbox


logger = logging.getLogger("preprocess")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default="./datasets/ocrtoc/3d_dataset",
        help="Path to OCRTOC dataset",
    )
    parser.add_argument(
        "--splits_file",
        type=Path,
        default="./datasets/ocrtoc/ocrtoc_splits_val05.json",
        help="Path to the splits file. This is used for final checking",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main(args):
    setup_logger(name="preprocess")

    with open(str(args.splits_file), "r") as f:
        splits = json.load(f)

    for split_name, split in splits.items():
        for scene_data in split:
            scene_name = scene_data["scene"]
            image_idss = scene_data["image_ids"]
            logger.info(f"Processing {scene_name}...")
            handle_model(args, scene_name, image_idss)


def handle_model(args, scene_name, image_idds):
    scene_dir = args.dataset_root / "scenes" / scene_name
    voxel_dir = scene_dir / "meshmvs_voxels"
    voxel_path = voxel_dir / "voxels_world.binvox"

    if not scene_dir.is_dir() \
            or not voxel_dir.is_dir() \
            or not voxel_path.is_file():
        return None

    # WARNING: Here we hardcode the assumption that the input voxels are
    # 128 x 128 x 128.
    with open(str(voxel_path), "rb") as f:
        # Read voxel coordinates as a tensor of shape (N, 3)
        voxel_coords = read_binvox_coords(f)

    camera_parameters = MeshVoxDataset.read_camera_parameters(
        str(args.dataset_root), scene_name
    )
    extrinsics = camera_parameters["extrinsics"]
    intrinsic = camera_parameters["intrinsic"]
    object_name = camera_parameters["object_name"]

    verts, faces = MeshVoxDataset.read_mesh(
        args.dataset_root, object_name, torch.eye(4)
    )

    # Align voxels to the same coordinate system as mesh verts, and save voxels.pt
    voxel_coords = align_bbox(voxel_coords, verts)
    voxel_data = {"voxel_coords": voxel_coords}
    voxel_path = voxel_dir / "voxels.pt"
    torch.save(voxel_data, voxel_path)

    # flatten image IDss
    # image_ids = set([j for i in image_idds for j in i])
    image_ids = list(extrinsics.keys())
    voxel_coords = voxel_coords.cuda()

    for idx, image_id in tqdm.tqdm(enumerate(image_ids), total=len(image_ids)):
        voxel_path = voxel_dir / ("vox_%03d.pt" % image_id)
        if voxel_path.is_file():
            continue

        extrinsic = extrinsics[image_id].cuda()
        voxel_coords_local = transform_verts(
            voxel_coords.unsqueeze(0), extrinsic.unsqueeze(0)
        )
        # non-homogeneous
        voxel_coords_local = voxel_coords_local[0, :, :3]

        # 48 x 48 x 48 voxels
        V = 48
        voxels = voxelize(voxel_coords_local, V).cpu()
        torch.save(voxels, voxel_path)

        # debug_voxels(voxels, voxel_coords, extrinsic, verts, faces)
        # exit(0)


def voxelize(voxel_coords, voxel_size):
    norm_coords = voxel_grid_coords([voxel_size]*3)
    grid_points = voxel_coords_to_world(
        norm_coords.view(-1, 3)
    ).view(1, -1, 3).contiguous().to(voxel_coords)
    voxel_coords = voxel_coords.unsqueeze(0).contiguous()

    depth_vox_nn = knn_points(grid_points, voxel_coords, K=1)

    voxel_width = (SHAPENET_MAX_ZMAX - SHAPENET_MIN_ZMIN) / float(voxel_size)
    voxel_width_square = voxel_width ** 2
    depth_vox_positive = depth_vox_nn.dists.view(1, *([voxel_size]*3)) \
                        < (voxel_width_square*8)
    return depth_vox_positive.squeeze(0)


def debug_voxels(voxels, voxel_coords, T_cam_world, verts, faces):
    T_cam_world = T_cam_world.cpu().numpy()
    print("non zero voxs", (voxels != 0).float().sum().item())
    # debug
    import open3d as o3d
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.cpu().numpy()),
        o3d.utility.Vector3iVector(faces.cpu().numpy())
    )
    o3d_mesh = o3d_mesh.transform(T_cam_world)

    from shapenet.modeling.voxel_ops import cubify
    cubified = cubify(
        logit(voxels + 1e-5).unsqueeze(0).float(), 64, 0.2
    )
    cubified_vertices = cubified[0].verts_packed().detach().cpu().numpy()
    cubified_faces = cubified[0].faces_packed().detach().cpu().numpy()

    cubified_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cubified_vertices),
        o3d.utility.Vector3iVector(cubified_faces),
    )

    voxels_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(voxel_coords.cpu().numpy())
    ).transform(T_cam_world)

    o3d.visualization.draw_geometries([
        o3d_mesh, voxels_pcd, cubified_o3d,
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    ])
    o3d.io.write_triangle_mesh("cubified.ply", cubified_o3d)
    o3d.io.write_triangle_mesh("gt.ply", o3d_mesh)
    o3d.io.write_point_cloud("voxels_pcd.ply", voxels_pcd)


if __name__ == "__main__":
    args = parse_args()
    main(args)
