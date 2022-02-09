import json
import logging
import os
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import Dataset

import torchvision.transforms as T
from PIL import Image
from shapenet.data.utils import imagenet_preprocess
from shapenet.utils.coords import SHAPENET_MAX_ZMAX, SHAPENET_MIN_ZMIN, project_verts
from .mesh_vox import MeshVoxDataset

logger = logging.getLogger(__name__)


class MeshVoxMultiViewDataset(MeshVoxDataset):
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
    ):
        # call the PyTorch Dataset interface in this way
        # since the immediate parent is MeshVoxDataset
        Dataset.__init__(self)
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
        self.mid_to_samples = {}
        # TODO: get the image ids from parameters
        # self.image_ids = input_views

        self.transform = self.get_transform(normalize_images)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]
        image_ids = self.image_ids[idx]

        metadata = self.read_camera_parameters(self.data_dir, scene_name)
        K = metadata["intrinsic"]

        imgs = []
        extrinsics = []
        for iid in image_ids:
            img = self.read_image(self.data_dir, scene_name, iid)
            img = self.transform(img)
            imgs.append(img)
            extrinsics.append(metadata["extrinsics"][iid])

        imgs = torch.stack(imgs, dim=0)
        extrinsics = torch.stack(extrinsics, dim=0)
        RT = extrinsics[0]

        # Maybe read mesh
        verts, faces = None, None
        if self.return_mesh:
            verts, faces = self.read_mesh(self.data_dir, metadata["object_name"], RT)

        # Maybe use cached samples
        points, normals = None, None
        if not self.sample_online:
            points, normals = self.sample_points_normals(
                self.data_dir, scene_name, RT
            )

        if self.voxel_size > 0:
            voxels = []
            for idx, iid in enumerate(image_ids):
                voxels.append(self.read_voxels(
                    self.data_dir, scene_name, iid, K, extrinsics[idx]
                ))
            voxels = torch.stack(voxels, dim=0)
        else:
            voxels = None

        P = K.mm(RT)

        id_str = "%s-%s" % (scene_name, '_'.join([str(i) for i in image_ids]))
        return {
            "imgs": imgs, "verts": verts, "faces": faces, "points": points,
            "normals": normals, "Ps": P, "id_str": id_str,
            "intrinsics": K, "extrinsics": extrinsics,
            "voxels": voxels,
        }
