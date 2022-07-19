import argparse
import sys
import os
from pathlib import Path
import glob
import re
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from test_idr import main as single_main


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("idr_dataset_root", type=Path)
    parser.add_argument("ocrtoc_dataset_root", type=Path)
    return parser.parse_args()


def get_scene_category_dict(ocrtoc_dataset_root):
    categories_file = ocrtoc_dataset_root / 'categories.json'
    with open(categories_file, 'r') as f:
        category_dict = json.load(f)

    scene_dict = {}
    for category, scenes in category_dict.items():
        for scene in scenes:
            scene_dict[scene] = category

    return category_dict, scene_dict


def main(args):
    pred_meshes = glob.glob(str(args.exp_dir) + "/**/ours_rgb/vis/000.ply", recursive=True)

    category_dict, scene_category_dict = get_scene_category_dict(args.ocrtoc_dataset_root)
    category_res = {k: {} for k in category_dict.keys()}

    for mesh_idx, pred_mesh in enumerate(pred_meshes):
        scene_name = pred_mesh.split("/")[-4]

        if scene_name.endswith("_64"):
            scene_name = scene_name[:-3]

        category = scene_category_dict[scene_name]

        cameras_path = args.idr_dataset_root / scene_name / scene_name / "cameras.npz"

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
        print(scene_name, category, pred_mesh, object_mesh)

        res = single_main(mesh_gt=object_mesh, mesh_pred=pred_mesh, cameras=cameras_path)
        category_res[category][scene_name] = (res["F1@0.300000"], res["Chamfer-L2"])

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

