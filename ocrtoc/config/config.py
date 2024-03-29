# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from fvcore.common.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
def get_ocrtoc_cfg():

    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.VOXEL_ON = False
    cfg.MODEL.MESH_ON = False
    cfg.MODEL.LIMIT_MESH_SIZE = True
    # options: predicted_depth_only | rendered_depth_only | input_concat | input_diff | feature_concat | feature_diff
    cfg.MODEL.CONTRASTIVE_DEPTH_TYPE = "input_concat"
    cfg.MODEL.USE_GT_DEPTH = False
    # options: multihead_attention | simple_attention | stats
    cfg.MODEL.FEATURE_FUSION_METHOD = "multihead_attention"
    cfg.MODEL.MULTIHEAD_ATTENTION =CN()
    # -1 maintains same feature dimensions as before attention
    cfg.MODEL.MULTIHEAD_ATTENTION.FEATURE_DIMS = 960
    cfg.MODEL.MULTIHEAD_ATTENTION.NUM_HEADS = 10

    # ------------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------------ #
    cfg.MODEL.CHECKPOINT = ""  # path to checkpoint

    # ------------------------------------------------------------------------ #
    # Voxel Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.VOXEL_HEAD = CN()
    # The number of convs in the voxel head and the number of channels
    cfg.MODEL.VOXEL_HEAD.NUM_CONV = 0
    cfg.MODEL.VOXEL_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers. Options: "" (no norm), "GN"
    cfg.MODEL.VOXEL_HEAD.NORM = ""
    # The number of depth channels for the predicted voxels
    cfg.MODEL.VOXEL_HEAD.VOXEL_SIZE = 28
    cfg.MODEL.VOXEL_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.VOXEL_HEAD.CUBIFY_THRESH = 0.0
    # voxel only iterations
    cfg.MODEL.VOXEL_HEAD.VOXEL_ONLY_ITERS = 100
    # Whether voxel weights are frozen
    cfg.MODEL.VOXEL_HEAD.FREEZE = False
    # Whether to use single view voxel prediction
    # without probabilistic merging
    cfg.MODEL.VOXEL_HEAD.SINGLE_VIEW = False
    cfg.MODEL.VOXEL_HEAD.RGB_FEATURES_INPUT = True
    cfg.MODEL.VOXEL_HEAD.DEPTH_FEATURES_INPUT = True
    cfg.MODEL.VOXEL_HEAD.RGB_BACKBONE = "resnet50"
    cfg.MODEL.VOXEL_HEAD.DEPTH_BACKBONE = "vgg"
    cfg.MODEL.VOXEL_HEAD.NOISE_FILTERING = False
    cfg.MODEL.VOXEL_HEAD.NOISE_FILTER_SIZE = 3
    # threshold in fraction, range [0.0, 1.0]
    cfg.MODEL.VOXEL_HEAD.NOISE_FILTER_THRESHOLD = 0.33

    # ------------------------------------------------------------------------ #
    # Voxel Refine Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.VOXEL_REFINE_HEAD = CN()
    cfg.MODEL.VOXEL_REFINE_HEAD.NUM_LAYERS = 2
    cfg.MODEL.VOXEL_REFINE_HEAD.VOXEL_ONLY = False

    # ------------------------------------------------------------------------ #
    # Mesh Head
    # ------------------------------------------------------------------------ #
    cfg.MODEL.MESH_HEAD = CN()
    cfg.MODEL.MESH_HEAD.NAME = "VoxMeshHead"
    # Numer of stages
    cfg.MODEL.MESH_HEAD.NUM_STAGES = 1
    cfg.MODEL.MESH_HEAD.NUM_GRAPH_CONVS = 1  # per stage
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_DIM = 256
    cfg.MODEL.MESH_HEAD.GRAPH_CONV_INIT = "normal"
    # Mesh sampling
    cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES = 5000
    cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES = 5000
    # whether to upsample mesh for training
    cfg.MODEL.MESH_HEAD.UPSAMPLE_PRED_MESH = True
    # loss weights
    cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT = 1.0
    cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT = 1.0
    # Init ico_sphere level (only for when voxel_on is false)
    cfg.MODEL.MESH_HEAD.ICO_SPHERE_LEVEL = -1

    cfg.MODEL.MESH_HEAD.RGB_FEATURES_INPUT = True
    cfg.MODEL.MESH_HEAD.DEPTH_FEATURES_INPUT = True
    cfg.MODEL.MESH_HEAD.RGB_BACKBONE = "resnet50"
    cfg.MODEL.MESH_HEAD.DEPTH_BACKBONE = "vgg"

    cfg.MODEL.MVSNET = CN()
    cfg.MODEL.MVSNET.FEATURES_LIST = [32, 64, 128, 256]
    cfg.MODEL.MVSNET.CHECKPOINT = ""
    cfg.MODEL.MVSNET.FREEZE = False

    # min/max depth: 0.60492384 2.1942575
    cfg.MODEL.MVSNET.MIN_DEPTH = 0.604
    cfg.MODEL.MVSNET.DEPTH_INTERVAL = 0.034
    cfg.MODEL.MVSNET.DEPTH_REFINE = True

    cfg.MODEL.MVSNET.NUM_DEPTHS = 48
    cfg.MODEL.MVSNET.INPUT_IMAGE_SIZE = (224, 224)
    cfg.MODEL.MVSNET.FOCAL_LENGTH = (248.0, 248.0)
    cfg.MODEL.MVSNET.PRINCIPAL_POINT = (111.5, 111.5)
    # loss weights
    cfg.MODEL.MVSNET.PRED_DEPTH_WEIGHT = 0.1
    cfg.MODEL.MVSNET.RENDERED_DEPTH_WEIGHT = 0.00
    cfg.MODEL.MVSNET.RENDERED_VS_GT_DEPTH_WEIGHT = 0.00

    # ------------------------------------------------------------------------ #
    # Solver
    # ------------------------------------------------------------------------ #
    cfg.SOLVER = CN()
    cfg.SOLVER.LR_SCHEDULER_NAME = "constant"  # {'constant', 'cosine'}
    cfg.SOLVER.BATCH_SIZE = 32
    cfg.SOLVER.BATCH_SIZE_EVAL = 8
    cfg.SOLVER.NUM_EPOCHS = 25
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.OPTIMIZER = "adam"  # {'sgd', 'adam'}
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_FACTOR = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 24949  # in iters
    cfg.SOLVER.LOGGING_PERIOD = 50  # in iters
    # stable training
    cfg.SOLVER.SKIP_LOSS_THRESH = 50.0
    cfg.SOLVER.LOSS_SKIP_GAMMA = 0.9
    # for saving checkpoint
    cfg.SOLVER.EARLY_STOP_METRIC = "F1@0.300000"

    # ------------------------------------------------------------------------ #
    # Datasets
    # ------------------------------------------------------------------------ #
    cfg.DATASETS = CN()
    cfg.DATASETS.NAME = "ocrtoc"
    # ['depth', 'multi_view', 'single_view']
    cfg.DATASETS.TYPE = "single_view"
    cfg.DATASETS.SPLITS_FILE = "./datasets/ocrtoc/ocrtoc_splits_val05.json"

    cfg.DATASETS.INPUT_VIEWS = [1, 35, 593]

    # ------------------------------------------------------------------------ #
    # Misc options
    # ------------------------------------------------------------------------ #
    # Directory where output files are written
    cfg.OUTPUT_DIR = "./output"

    return cfg

