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

from test_colmap import main as single_main
from test_dvr_all import get_scene_category_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("idr_dataset_root", type=Path)
    parser.add_argument("ocrtoc_dataset_root", type=Path)
    return parser.parse_args()


def main(args):
    category_dict, scene_category_dict = get_scene_category_dict(args.ocrtoc_dataset_root)
    category_res = {k: {} for k in category_dict.keys()}

    for pred_idx, pred_dir in enumerate(args.exp_dir.iterdir()):
        res = single_main(
            pred_dir, args.idr_dataset_root, args.ocrtoc_dataset_root
        )
        if res is None:
            continue

        # print(res["scene_name"], res["category"])
        category = res["category"]
        scene_name = res["scene_name"]
        metrics = res["metrics"]

        category_res[category][scene_name+"_"+str(pred_idx)] = (
            metrics["F1@0.300000"], metrics["Chamfer-L2"]
        )

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
