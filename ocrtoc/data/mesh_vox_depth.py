import json
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image
import cv2
from shapenet.data.utils import imagenet_preprocess
from shapenet.utils.coords import SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN, project_verts
from .mesh_vox_multi_view import MeshVoxMultiViewDataset

logger = logging.getLogger(__name__)

# 0.57 is the scaling used by the 3D-R2N2 dataset
# 1000 is the scale applied for saving depths as ints
DEPTH_SCALE = 1000


class MeshVoxDepthDataset(MeshVoxMultiViewDataset):
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
        input_views=[0, 6, 7],
        depth_only=False,
    ):
        MeshVoxMultiViewDataset.__init__(
            self, data_dir, normalize_images=normalize_images,
            split=split, return_mesh=return_mesh, voxel_size=voxel_size,
            num_samples=num_samples, sample_online=sample_online,
            in_memory=in_memory, return_id_str=return_id_str,
            input_views=input_views
        )

        self.set_depth_only(depth_only)

        self.scene_names = []
        self.image_ids = []
        for scene_data in split:
            for image_ids in scene_data["image_ids"]:
                self.scene_names.append(scene_data["scene"])
                self.image_ids.append(image_ids)


    def set_depth_only(self, value):
        self.depth_only = value

    @staticmethod
    def read_depth(data_dir, scene_name, iid):
        depth_file = os.path.join(
            data_dir, scene_name,
            "meshmvs_training_images", "depth_" + str(iid) + ".png"
        )
        if os.path.isfile(depth_file):
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / DEPTH_SCALE
            depth = torch.from_numpy(depth)
            return depth
        else:
            print('depth file not found:', depth_file)
            exit(1)

    @staticmethod
    def read_mask(data_dir, scene_name, iid):
        mask_file = os.path.join(
            data_dir, scene_name,
            "gt_masks", "mask_" + str(iid) + ".png"
        )
        if os.path.isfile(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask)
            return mask
        else:
            print('mask file not found:', mask_file)
            exit(1)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]
        image_ids = self.image_ids[idx]

        metadata = self.read_camera_parameters(self.data_dir, scene_name)

        depths = []
        masks = []
        for iid in image_ids:
            depths.append(self.read_depth(self.data_dir, scene_name, iid))
            # masks.append(self.read_mask(self.data_dir, scene_name, iid))
            masks.append((depths[-1] > 1e-2).float())

        depths = torch.stack(depths, dim=0)
        masks = torch.stack(masks, dim=0)
        masks = F.interpolate(
            masks.view(-1, 1, *(masks.shape[1:])),
            depths.shape[-2:], mode="bilinear", align_corners=False
        ).view(*(depths.shape))

        if self.depth_only:
            # depths, masks, images and camera parameters
            K = metadata["intrinsic"]
            imgs = torch.stack([
                self.transform(self.read_image(
                    self.data_dir, scene_name, iid
                ))
                for iid in image_ids
            ], dim=0)

            extrinsics = torch.stack(
                [metadata["extrinsics"][iid] for iid in image_ids], dim=0
            )
            id_str = "%s-%s" % (scene_name, '_'.join([str(i) for i in image_ids]))
            return {
                "depths": depths, "masks": masks, "imgs": imgs,
                "intrinsics": K, "extrinsics": extrinsics,
                "id_str": id_str
            }
        else:
            return {
                **MeshVoxMultiViewDataset.__getitem__(self, idx),
                "depths": depths, "masks": masks
            }

