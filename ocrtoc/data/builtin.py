# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file registers pre-defined datasets at hard-coded paths
"""
import os

# each dataset contains name : (data_dir, splits_file)
_PREDEFINED_SPLITS_OCRTOC = {
    "ocrtoc": ("ocrtoc/3d_dataset", "ocrtoc/ocrtoc_splits_val05.json")
}


def register_ocrtoc(dataset_name, root="datasets"):
    if dataset_name not in _PREDEFINED_SPLITS_OCRTOC.keys():
        raise ValueError("%s not registered" % dataset_name)
    data_dir = _PREDEFINED_SPLITS_OCRTOC[dataset_name][0]
    return os.path.join(root, data_dir)
