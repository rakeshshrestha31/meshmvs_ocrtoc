# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import logging
import os
import pickle
import tempfile
import numpy as np
import open3d as o3d
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import torchvision.transforms as T
from PIL import Image
from shapenet.data.utils import imagenet_preprocess
from shapenet.utils.coords import (
    project_verts, world_coords_to_voxel, get_blender_intrinsic_matrix,
    SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN
)


logger = logging.getLogger(__name__)


class MeshVoxDataset(Dataset):
    def __init__(
        self,
        data_dir,
        normalize_images=True,
        split=None,
        return_mesh=False,
        voxel_size=32,
        num_samples=5000,
        sample_online=False,
        in_memory=False,
        return_id_str=False,
    ):

        super(MeshVoxDataset, self).__init__()
        if not return_mesh and sample_online:
            raise ValueError("Cannot sample online without returning mesh")
        self.data_dir = data_dir
        self.return_mesh = return_mesh
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.sample_online = sample_online
        self.return_id_str = return_id_str

        self.synset_ids = []
        self.model_ids = []
        self.image_ids = []
        self.mid_to_samples = {}

        self.transform = self.get_transform(normalize_images)

        summary_json = os.path.join(data_dir, "summary.json")
        with open(summary_json, "r") as f:
            summary = json.load(f)
            for sid in summary:
                logger.info("Starting synset %s" % sid)
                allowed_mids = None
                if split is not None:
                    if sid not in split:
                        logger.info("Skipping synset %s" % sid)
                        continue
                    elif isinstance(split[sid], list):
                        allowed_mids = set(split[sid])
                    elif isinstance(split, dict):
                        allowed_mids = set(split[sid].keys())
                for mid, num_imgs in summary[sid].items():
                    if allowed_mids is not None and mid not in allowed_mids:
                        continue
                    allowed_iids = None
                    if split is not None and isinstance(split[sid], dict):
                        allowed_iids = set(split[sid][mid])
                    if not sample_online and in_memory:
                        samples_path = os.path.join(data_dir, sid, mid, "samples.pt")
                        samples = torch.load(samples_path)
                        self.mid_to_samples[mid] = samples
                    for iid in range(num_imgs):
                        if allowed_iids is None or iid in allowed_iids:
                            self.synset_ids.append(sid)
                            self.model_ids.append(mid)
                            self.image_ids.append(iid)

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def get_transform(normalize_images):
        transform = [T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        return T.Compose(transform)

    @staticmethod
    def read_camera_parameters(data_dir, scene_name):
        camera_poses = np.load(
            os.path.join(data_dir, scene_name, "meshmvs_camera_poses.npy"),
            allow_pickle=True
        ).item()

        object_poses = np.load(
            os.path.join(data_dir, scene_name, "meshmvs_object_poses.npy"),
            allow_pickle=True
        ).item()
        object_name, object_pose = next(iter(object_poses.items()))

        extrinsics = MeshVoxDataset.process_extrinsics(camera_poses, object_pose)

        K = np.load(
            os.path.join(data_dir, scene_name, "meshmvs_camK.npy"),
        ).reshape((3, 3)).astype(np.float32)
        K = MeshVoxDataset.normalize_intrinsics(K)
        # K = torch.from_numpy(K)

        return {
            "extrinsics": extrinsics, "intrinsic": K, "object_name": object_name
        }

    @staticmethod
    def normalize_intrinsics(K):
        return get_blender_intrinsic_matrix()

    @staticmethod
    def process_extrinsics(camera_poses, object_pose):
        extrinsics = {}

        T_EUS_EDN = np.asarray([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0]
        ])
        T_EDN_EUS = T_EUS_EDN.T

        T_EDN_obj = object_pose
        T_obj_EDN = np.linalg.inv(T_EDN_obj)

        # object centric pose, EUS oriented camera
        for cam_idx, T_EDN_cam in camera_poses.items():
            T_obj_cam = T_obj_EDN @ T_EDN_cam
            # to EUS from EDN
            T_obj_cam = T_obj_cam @ T_EDN_EUS
            T_cam_obj = np.linalg.inv(T_obj_cam)
            extrinsics[cam_idx] = torch.from_numpy(T_cam_obj.astype(np.float32))

        return extrinsics

    @staticmethod
    def read_image(data_dir, scene_name, iid):
        img_path = os.path.join(
            data_dir, scene_name,
            "meshmvs_training_images", f"color_{iid}.png"
        )
        # Load the image
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        return img

    @staticmethod
    def read_mesh(data_dir, scene_name, RT):
        mesh_path = os.path.join(data_dir, scene_name, "meshmvs_model.obj")
        o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_mesh_path = os.path.join(tmp, 'tmp.obj')
            o3d.io.write_triangle_mesh(tmp_mesh_path, o3d_mesh)
            verts, faces, _ = load_obj(tmp_mesh_path, load_textures=False)
        faces = faces.verts_idx
        verts = project_verts(verts, RT)
        return verts, faces

    def read_voxels(self, data_dir, scene_name, iid, K, RT):
        # Use precomputed voxels if we have them, otherwise return voxel_coords
        # and we will compute voxels in postprocess
        voxel_dir = os.path.join(
            data_dir, scene_name, "meshmvs_voxels"
        )
        voxel_file = os.path.join(voxel_dir, "vox_%03d.pt" % iid)
        P = None

        if os.path.isfile(voxel_file):
            voxels = torch.load(voxel_file)
        else:
            voxel_path = os.path.join(voxel_dir, "voxels.pt")
            voxel_data = torch.load(voxel_path)
            voxels = voxel_data["voxel_coords"]
            P = K.mm(RT)
        return voxels

    def sample_points_normals(self, data_dir, scene_name, RT):
        samples = self.mid_to_samples.get(scene_name, None)
        if samples is None:
            # They were not cached in memory, so read off disk
            samples_path = os.path.join(
                data_dir, scene_name, "meshmvs_gt_labels.dat"
            )
            with open(samples_path, 'rb') as f:
                samples = pickle.load(f)
        points = torch.from_numpy(samples[:, :3])
        normals = torch.from_numpy(samples[:, 3:6])
        idx = torch.randperm(points.shape[0])[: self.num_samples]
        points, normals = points[idx], normals[idx]
        points = project_verts(points, RT)
        normals = normals.mm(RT[:3, :3].t())  # Only rotate, don't translate
        return points, normals

    def __getitem__(self, idx):
        sid = self.synset_ids[idx]
        mid = self.model_ids[idx]
        iid = self.image_ids[idx]

        metadata = self.read_camera_parameters(self.data_dir, sid, mid)
        K = metadata["intrinsic"]
        RT = metadata["extrinsics"][iid]

        img_path = metadata["image_list"][iid]
        img = self.read_image(self.data_dir, sid, mid, img_path)
        img = self.transform(img)

        # Maybe read mesh
        verts, faces = None, None
        if self.return_mesh:
            verts, faces = self.read_mesh(self.data_dir, sid, mid, RT)

        # Maybe use cached samples
        points, normals = None, None
        if not self.sample_online:
            points, normals = self.sample_points_normals(
                self.data_dir, sid, mid, RT
            )

        voxels, P = None, None
        if self.voxel_size > 0:
            voxels, P = self.read_voxels(self.data_dir, sid, mid, iid, K, RT)

        id_str = "%s-%s-%02d" % (sid, mid, iid)
        # add dim=1 for view (single-view)
        return {
            "imgs": imgs, "verts": verts, "faces": faces, "points": points,
            "voxels": normals, "voxels": voxels,
            "Ps": P, "intrinsics": K, "extrinsics": RT.unsqueeze(1),
            "id_str": id_str
        }

    def _voxelize(self, voxel_coords, P):
        V = self.voxel_size
        device = voxel_coords.device
        voxel_coords = world_coords_to_voxel(
            voxel_coords.unsqueeze(0), P.unsqueeze(0)
        ).squeeze(0)

        # Now voxels are in [-1, 1]^3; map to [0, V-1)^3
        voxel_coords = 0.5 * (V - 1) * (voxel_coords + 1.0)
        voxel_coords = voxel_coords.round().to(torch.int64)
        valid = (0 <= voxel_coords) * (voxel_coords < V)
        valid = valid[:, 0] * valid[:, 1] * valid[:, 2]
        x, y, z = voxel_coords.unbind(dim=1)
        x, y, z = x[valid], y[valid], z[valid]
        voxels = torch.zeros(V, V, V, dtype=torch.int64, device=device)
        voxels[z, y, x] = 1

        return voxels

    @staticmethod
    def collate_fn(batch):
        """
        Args:
        - batch: list of dicts
        Returns:
        - dicts with collated items
        """
        assert(len(batch))

        # these need special treatment
        non_standard_collate_keys = [
            "verts", "faces", "points", "normals", "voxels", "Ps", "id_str"
        ]
        standard_batch = [{
            key: value for key, value in batch_item.items()
            if key not in non_standard_collate_keys
        } for batch_item in batch]
        collated_batch = default_collate(standard_batch)

        def extract_key(batch, key):
            return [ i.get(key, None) for i in batch ]

        if batch[0].get("verts", None) is not None and batch[0].get("faces", None) is not None:
            # TODO(gkioxari) Meshes should accept tuples
            collated_batch["meshes"] = Meshes(
                verts=extract_key(batch, "verts"),
                faces=extract_key(batch, "faces")
            )
        else:
            collated_batch["meshes"] = None

        def none_safe_collate_fn(batch, key):
            """
            Simple collate with protection against None items
            """
            items = extract_key(batch, key)
            if None not in items:
                return torch.stack(items, dim=0)
            else:
                return None

        if "voxels" not in batch[0] or batch[0]["voxels"] is None:
            voxels = None
            Ps = None
        elif batch[0]["voxels"].dim() == 2:
            # They are voxel coords
            collated_batch["voxels"] = extract_key(batch, "voxels")
        elif batch[0]["voxels"].dim() == 4:
            # They are actual voxels
            collated_batch["voxels"] = none_safe_collate_fn(batch, "voxels")

        collated_batch["points"] = none_safe_collate_fn(batch, "points")
        collated_batch["normals"] = none_safe_collate_fn(batch, "normals")
        collated_batch["Ps"] = none_safe_collate_fn(batch, "Ps")
        collated_batch["id_strs"] = extract_key(batch, "id_str")

        return collated_batch

    def postprocess(self, batch, device=None):
        if device is None:
            device = torch.device("cuda")
        non_standard_keys = ["points", "normals", "voxels", "id_strs"]

        # process standard items
        processed_batch = {
            key: (value.to(device) if value is not None else None)
            for key, value in batch.items()
            if key not in non_standard_keys
        }

        # process non-standard items
        if batch["points"] is not None and batch["normals"] is not None:
            processed_batch["points"] = batch["points"].to(device)
            processed_batch["normals"] = batch["normals"].to(device)
        elif batch.get("meshes", None) is not None:
            processed_batch["points"], processed_batch["normals"] = \
                sample_points_from_meshes(
                    batch["meshes"], num_samples=self.num_samples,
                    return_normals=True
                )
        if "voxels" in batch and batch["voxels"] is not None:
            if torch.is_tensor(batch["voxels"]):
                # We used cached voxels on disk, just cast and return
                processed_batch["voxels"] = batch["voxels"].to(device)
                # TODO: need to transform voxel grid to all views
                # raise NotImplementedError(
                #     "need to transform voxel grid to all views"
                # )
            else:
                # We got a list of voxel_coords, and need to compute voxels on-the-fly
                voxel_coords = batch["voxels"]
                voxels = []
                for batch_idx, cur_voxel_coords in enumerate(voxel_coords):
                    cur_voxel_coords = cur_voxel_coords.to(device)
                    voxels_views = []
                    K = batch["intrinsics"][batch_idx]
                    # find voxel grid in all views coordinate frames
                    for view_idx, transform in \
                            enumerate(batch["extrinsics"][batch_idx].unbind(0)):
                        P = K.matmul(transform)
                        cur_voxels = self._voxelize(cur_voxel_coords, P)
                        voxels_views.append(cur_voxels)
                    voxels.append(torch.stack(voxels_views, dim=0))
                processed_batch["voxels"] = torch.stack(voxels, dim=0)

        if self.return_id_str:
            processed_batch["id_strs"] = batch["id_strs"]

        return processed_batch
