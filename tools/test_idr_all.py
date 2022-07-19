#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path
import glob
import re
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from test_idr import main as single_main
from test_dvr_all import get_scene_category_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("idr_dataset_root", type=Path)
    parser.add_argument("ocrtoc_dataset_root", type=Path)
    return parser.parse_args()


def main(args):
    pred_meshes = glob.glob(str(args.exp_dir) + "/**/" + "surface_1400.ply", recursive=True)

    category_dict, scene_category_dict = get_scene_category_dict(args.ocrtoc_dataset_root)
    category_res = {k: {} for k in category_dict.keys()}

    for mesh_idx, pred_mesh in enumerate(pred_meshes):
        scan_id = re.findall("orctoc_fixed_cameras_\d+", pred_mesh)[0].split("_")[-1]
        cameras_path = args.idr_dataset_root / f"scan{scan_id}" / "cameras.npz"
        scene_name = os.path.realpath(str(args.idr_dataset_root / f"scan{scan_id}")).split("/")[-1]
        category = scene_category_dict[scene_name]

        object_files = glob.glob(
            str(args.ocrtoc_dataset_root) + f"/scenes/**/{scene_name}/object_list.txt", recursive=True
        )
        if not object_files:
            print("object list not found. skipping", pred_mesh)
            continue

        object_file = object_files[0]

        with open(object_file, "r") as f:
            object_name = f.read().strip()

        object_model_dirs = glob.glob(
            str(args.ocrtoc_dataset_root) + f"/models/**/{object_name}", recursive=True
        )

        if not object_model_dirs:
            print("object model not found. skipping", pred_mesh)
            continue

        object_model_dir = object_model_dirs[0]

        object_mesh = Path(object_model_dir) / "visual.ply"
        print(scene_name, category, scan_id, pred_mesh, object_mesh)

        res = single_main(mesh_gt=object_mesh, mesh_pred=pred_mesh, cameras=cameras_path)
        category_res[category][scene_name+str(mesh_idx)] = (res["F1@0.300000"], res["Chamfer-L2"])

    means = {
        category: {
            "means": [np.mean(i) for i in zip(*(res.values()))],
            **res
        }
        for category, res in category_res.items()
    }
    print(json.dumps(means, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)
