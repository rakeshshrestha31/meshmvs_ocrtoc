#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import open3d as o3d
import logging
import os
import shutil
import time
import numpy as np
import tqdm
import json
import gc
import traceback

import detectron2.utils.comm as comm
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import save_obj
from pytorch3d.transforms import Transform3d

from shapenet.config import get_shapenet_cfg
from shapenet.data import build_data_loader, register_shapenet
from shapenet.evaluation import \
        evaluate_split, evaluate_test, evaluate_test_p2m, evaluate_vox
from shapenet.utils.coords import relative_extrinsics, \
        get_blender_intrinsic_matrix

# required so that .register() calls are executed in module scope
from shapenet.modeling import MeshLoss, build_model
from shapenet.modeling.heads.depth_loss import adaptive_berhu_loss
from shapenet.modeling.mesh_arch import VoxMeshMultiViewHead, VoxMeshDepthHead
from shapenet.solver import build_lr_scheduler, build_optimizer
from shapenet.utils import Checkpoint, Timer, clean_state_dict, default_argument_parser
from shapenet.utils.depth_backprojection import get_points_from_depths
from meshrcnn.utils.metrics import compare_meshes

logger = logging.getLogger("shapenet")
P2M_SCALE = 0.57
NUM_PRED_SURFACE_SAMPLES = 6466
NUM_GT_SURFACE_SAMPLES = 9000


def copy_data(args):
    data_base, data_ext = os.path.splitext(os.path.basename(args.data_dir))
    assert data_ext in [".tar", ".zip"]
    t0 = time.time()
    logger.info("Copying %s to %s ..." % (args.data_dir, args.tmp_dir))
    data_tmp = shutil.copy(args.data_dir, args.tmp_dir)
    t1 = time.time()
    logger.info("Copying took %fs" % (t1 - t0))
    logger.info("Unpacking %s ..." % data_tmp)
    shutil.unpack_archive(data_tmp, args.tmp_dir)
    t2 = time.time()
    logger.info("Unpacking took %f" % (t2 - t1))
    args.data_dir = os.path.join(args.tmp_dir, data_base)
    logger.info("args.data_dir = %s" % args.data_dir)


def get_dataset_name(cfg):
    if cfg.DATASETS.TYPE.lower() == "multi_view":
        return "MeshVoxMultiView"
    elif cfg.DATASETS.TYPE.lower() == "depth":
        return "MeshVoxDepth"
    elif cfg.DATASETS.TYPE.lower() == "single_view":
        return "MeshVox"
    else:
        print("unrecognized dataset type", cfg.DATASETS.TYPE)
        exit(1)


def main_worker_eval(worker_id, args):

    device = torch.device("cuda:%d" % worker_id)
    cfg = setup(args)

    # build test set
    test_loader = build_data_loader(
        cfg, get_dataset_name(cfg), "test", multigpu=False
    )
    logger.info("test - %d" % len(test_loader))

    # load checkpoing and build model
    if cfg.MODEL.CHECKPOINT == "":
        raise ValueError("Invalid checkpoing provided")
    logger.info("Loading model from checkpoint: %s" % (cfg.MODEL.CHECKPOINT))
    cp = torch.load(PathManager.get_local_path(cfg.MODEL.CHECKPOINT))

    if args.eval_latest_checkpoint:
        logger.info("using latest checkpoint weights")
        state_dict = clean_state_dict(cp["latest_states"]["model"])
    else:
        logger.info("using best checkpoint weights")
        state_dict = clean_state_dict(cp["best_states"]["model"])

    model = build_model(cfg)
    model.load_state_dict(state_dict)
    logger.info("Model loaded")
    model.to(device)

    def disable_running_stats(model):
        if type(model).__name__.startswith("BatchNorm"):
            model.track_running_stats = False
        else:
            for m in model.children():
                disable_running_stats(m)
    # disable_running_stats(model)

    val_loader = build_data_loader(
        cfg, get_dataset_name(cfg), "test", multigpu=False
    )
    logger.info("val - %d" % len(val_loader))
    test_metrics, test_preds = evaluate_split(
        model, val_loader, prefix="val_", max_predictions=100
    )
    str_out = "Results on test"
    for k, v in test_metrics.items():
        str_out += "%s %.4f " % (k, v)
    logger.info(str_out)

    if args.eval_vox:
        logger.info("running eval_vox")
        prediction_dir = os.path.join(
            cfg.OUTPUT_DIR, "predictions"
        )
        test_metrics = evaluate_vox(model, test_loader, prediction_dir)
        print(test_metrics)
        str_out = "Results on test"
        for k, v in test_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)
    elif args.eval_p2m:
        logger.info("running eval_p2m")
        test_metrics = evaluate_test_p2m(model, test_loader)

        # model_type -> mesh, vox
        for model_type in test_metrics.keys():
            # model_type -> precisions, recalls, f_scores
            for metric_type in test_metrics[model_type].keys():
                # model_type -> sum, mean, object
                for metric_subtype in test_metrics[model_type][metric_type].keys():
                    file_path = os.path.join(
                        cfg.OUTPUT_DIR,
                        "%s_%s_%s.json" % (model_type, metric_type, metric_subtype)
                    )
                    metric = test_metrics[model_type][metric_type][metric_subtype]
                    with open(file_path, "w") as f:
                        json.dump({
                            key: value.tolist()
                            for key, value in metric.items()
                        }, f, indent=4)

    else:
        logger.info("saving predictions")
        prediction_dir = os.path.join(
            cfg.OUTPUT_DIR, "predictions", "eval", "predict"
        )
        if (not args.eval_save_point_clouds) and (not args.eval_save_meshes):
            logger.info("both save-point-clouds and save-meshes false")
            args.eval_save_point_clouds = True

        save_predictions(
            model, test_loader, prediction_dir,
            args.eval_save_point_clouds, args.eval_save_meshes,
            args.eval_save_initial_meshes
        )
    # else:
    #     evaluate_test(model, test_loader)


def main_worker(worker_id, args):
    distributed = False
    if args.num_gpus > 1:
        distributed = True
        dist.init_process_group(
            backend="NCCL", init_method=args.dist_url, world_size=args.num_gpus, rank=worker_id
        )
        torch.cuda.set_device(worker_id)

    device = torch.device("cuda:%d" % worker_id)

    cfg = setup(args)

    # data loaders
    loaders = setup_loaders(cfg)
    for split_name, loader in loaders.items():
        logger.info("%s - %d" % (split_name, len(loader)))

    # build the model
    model = build_model(cfg)
    model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[worker_id],
            output_device=worker_id,
            check_reduction=True,
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )

    optimizer = build_optimizer(cfg, model)
    cfg.SOLVER.COMPUTED_MAX_ITERS = cfg.SOLVER.NUM_EPOCHS * len(loaders["train"])
    scheduler = build_lr_scheduler(cfg, optimizer)

    loss_fn_kwargs = {
        "chamfer_weight": cfg.MODEL.MESH_HEAD.CHAMFER_LOSS_WEIGHT,
        "normal_weight": cfg.MODEL.MESH_HEAD.NORMALS_LOSS_WEIGHT,
        "edge_weight": cfg.MODEL.MESH_HEAD.EDGE_LOSS_WEIGHT,
        "voxel_weight": cfg.MODEL.VOXEL_HEAD.LOSS_WEIGHT,
        "gt_num_samples": cfg.MODEL.MESH_HEAD.GT_NUM_SAMPLES,
        "pred_num_samples": cfg.MODEL.MESH_HEAD.PRED_NUM_SAMPLES,
        "upsample_pred_mesh": cfg.MODEL.MESH_HEAD.UPSAMPLE_PRED_MESH,
    }
    loss_fn = MeshLoss(**loss_fn_kwargs)

    checkpoint_path = "checkpoint.pt"
    checkpoint_path = os.path.join(cfg.OUTPUT_DIR, checkpoint_path)
    cp = Checkpoint(checkpoint_path)
    if len(cp.restarts) == 0:
        # We are starting from scratch, so store some initial data in cp
        iter_per_epoch = len(loaders["train"])
        cp.store_data("iter_per_epoch", iter_per_epoch)
    else:
        logger.info("Loading model state from checkpoint")

        if args.train_best_checkpoint:
            logger.info("using best checkpoint weights")
            model.load_state_dict(cp.best_states["model"])
            optimizer.load_state_dict(cp.best_states["optim"])
            scheduler.load_state_dict(cp.best_states["lr_scheduler"])
        else:
            logger.info("using latest checkpoint weights")
            model.load_state_dict(cp.latest_states["model"])
            optimizer.load_state_dict(cp.latest_states["optim"])
            scheduler.load_state_dict(cp.latest_states["lr_scheduler"])

    training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn)


def training_loop(cfg, cp, model, optimizer, scheduler, loaders, device, loss_fn):
    Timer.timing = False
    iteration_timer = Timer("Iteration")

    # model.parameters() is surprisingly expensive at 150ms, so cache it
    if hasattr(model, "module"):
        params = list(model.module.parameters())
    else:
        params = list(model.parameters())
    loss_moving_average = cp.data.get("loss_moving_average", None)
    while cp.epoch < cfg.SOLVER.NUM_EPOCHS:
        if comm.is_main_process():
            logger.info("Starting epoch %d / %d" % (cp.epoch + 1, cfg.SOLVER.NUM_EPOCHS))

        # When using a DistributedSampler we need to manually set the epoch so that
        # the data is shuffled differently at each epoch
        for loader in loaders.values():
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(cp.epoch)

        for i, batch in enumerate(loaders["train"]):
            if i == 0:
                iteration_timer.start()
            else:
                iteration_timer.tick()

            batch = loaders["train"].postprocess(batch, device)

            num_infinite_params = 0
            for p in params:
                num_infinite_params += (torch.isfinite(p.data) == 0).sum().item()
            if num_infinite_params > 0:
                msg = "ERROR: Model has %d non-finite params (before forward!)"
                logger.info(msg % num_infinite_params)
                return

            model_kwargs = {}
            if cfg.MODEL.VOXEL_ON and cp.t < cfg.MODEL.VOXEL_HEAD.VOXEL_ONLY_ITERS:
                model_kwargs["voxel_only"] = True

            module = model.module if hasattr(model, "module") else model
            if isinstance(module, VoxMeshMultiViewHead):
                model_kwargs["intrinsics"] = batch["intrinsics"]
                model_kwargs["extrinsics"] = batch["extrinsics"]
            if isinstance(module, VoxMeshDepthHead):
                model_kwargs["masks"] = batch["masks"]
                if cfg.MODEL.USE_GT_DEPTH:
                    model_kwargs["depths"] = batch["depths"]

            with Timer("Forward"):
                model_outputs = model(batch["imgs"], **model_kwargs)
                voxel_scores = model_outputs.get("voxel_scores", None)
                meshes_pred = model_outputs.get("meshes_pred", [])
                merged_voxel_scores = model_outputs.get(
                    "merged_voxel_scores", None
                )

            # debug only
            # if len(meshes_pred) > 0:
            #     mean_V = meshes_pred[-1].num_verts_per_mesh().tolist()
            #     mean_F = meshes_pred[-1].num_faces_per_mesh().tolist()
            # logger.info("mesh size = (%r)" % (list(zip(mean_V, mean_F))))

            num_infinite = 0
            for cur_meshes in meshes_pred:
                cur_verts = cur_meshes.verts_packed()
                num_infinite += (torch.isfinite(cur_verts) == 0).sum().item()
            if num_infinite > 0:
                logger.info("ERROR: Got %d non-finite verts" % num_infinite)
                return

            if num_infinite == 0:
                loss, losses = loss_fn(
                    voxel_scores, merged_voxel_scores,
                    meshes_pred, batch["voxels"],
                    (batch["points"], batch["normals"])
                )

            skip = loss is None
            if loss is None or (torch.isfinite(loss) == 0).sum().item() > 0:
                logger.info("WARNING: Got non-finite loss %f" % loss)
                skip = True
            # depth losses
            elif "depths" in batch:
                if not cfg.MODEL.USE_GT_DEPTH:
                    if "pred_depths" in model_outputs:
                        depth_loss = adaptive_berhu_loss(
                            batch["depths"], model_outputs["pred_depths"],
                            batch["masks"]
                        )
                        if not torch.any(torch.isnan(depth_loss)):
                            loss = loss \
                                 + (depth_loss * cfg.MODEL.MVSNET.PRED_DEPTH_WEIGHT)
                        else:
                            logger.info("WARNING: Got NaN depth loss")
                        losses["pred_depth_loss"] = depth_loss
                if "rendered_depths" in model_outputs \
                        and not model_kwargs.get("voxel_only", False):
                    pred_depths = model_outputs["pred_depths"]
                    masks = batch["masks"]
                    all_ones_masks = torch.ones_like(masks)
                    if pred_depths is not None:
                        resized_masks = F.interpolate(
                            masks.view(-1, 1, *(masks.shape[2:])),
                            pred_depths.shape[-2:], mode="nearest"
                        ).view(*(masks.shape[:2]), *(pred_depths.shape[-2:]))
                        masked_depths = pred_depths * resized_masks
                    else:
                        masked_depths = None

                    for depth_idx, rendered_depth in \
                            enumerate(model_outputs["rendered_depths"]):
                        rendered_depth_loss = adaptive_berhu_loss(
                            masked_depths, rendered_depth, all_ones_masks
                        )
                        if not torch.any(torch.isnan(rendered_depth_loss)):
                            loss = loss \
                                 + (rendered_depth_loss \
                                    * cfg.MODEL.MVSNET.RENDERED_DEPTH_WEIGHT)
                        losses["rendered_depth_loss_%d" % depth_idx] \
                                = rendered_depth_loss

                        # rendered vs GT depth loss, only for debug
                        rendered_gt_depth_loss = adaptive_berhu_loss(
                            batch["depths"], rendered_depth, all_ones_masks
                        )
                        losses["rendered_gt_depth_loss_%d" % depth_idx] \
                                = rendered_gt_depth_loss
                        if not torch.any(torch.isnan(rendered_gt_depth_loss)):
                            loss = loss \
                                 + (rendered_gt_depth_loss \
                                    * cfg.MODEL.MVSNET.RENDERED_VS_GT_DEPTH_WEIGHT)

            if model_kwargs.get("voxel_only", False):
                for k, v in losses.items():
                    if "voxel" not in k:
                        losses[k] = 0.0 * v

            if loss is not None and cp.t % cfg.SOLVER.LOGGING_PERIOD == 0:
                if comm.is_main_process():
                    cp.store_metric(loss=loss.item())
                    str_out = "Iteration: %d, epoch: %d, lr: %.5f," % (
                        cp.t,
                        cp.epoch,
                        optimizer.param_groups[0]["lr"],
                    )
                    for k, v in losses.items():
                        str_out += "  %s loss: %.4f," % (k, v.item())
                    str_out += "  total loss: %.4f," % loss.item()

                    # memory allocaged
                    if torch.cuda.is_available():
                        max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                        str_out += " mem: %d" % max_mem_mb

                    if len(meshes_pred) > 0:
                        mean_V = meshes_pred[-1].num_verts_per_mesh().float().mean().item()
                        mean_F = meshes_pred[-1].num_faces_per_mesh().float().mean().item()
                        str_out += ", mesh size = (%d, %d)" % (mean_V, mean_F)
                    logger.info(str_out)

            if loss_moving_average is None and loss is not None:
                loss_moving_average = loss.item()

            # Skip backprop for this batch if the loss is above the skip factor times
            # the moving average for losses
            if loss is None:
                pass
            elif loss.item() > cfg.SOLVER.SKIP_LOSS_THRESH * loss_moving_average:
                logger.info("Warning: Skipping loss %f on GPU %d" % (loss.item(), comm.get_rank()))
                cp.store_metric(losses_skipped=loss.item())
                skip = True
            else:
                # Update the moving average of our loss
                gamma = cfg.SOLVER.LOSS_SKIP_GAMMA
                loss_moving_average *= gamma
                loss_moving_average += (1.0 - gamma) * loss.item()
                cp.store_data("loss_moving_average", loss_moving_average)

            if skip:
                logger.info("Dummy backprop on GPU %d" % comm.get_rank())
                loss = 0.0 * sum(p.sum() for p in params)

            # Backprop and step
            scheduler.step()
            optimizer.zero_grad()

            try:
                with Timer("Backward"):
                    loss.backward()
            except RuntimeError as e:
                logger.info("Caught Runtime Error {}".format(e))
                traceback.print_exc()

                if "meshes_pred" in locals():
                    mean_V = meshes_pred[-1].num_verts_per_mesh().tolist()
                    mean_F = meshes_pred[-1].num_faces_per_mesh().tolist()
                    print("device: %r, mesh size = (%r)" % (device, list(zip(mean_V, mean_F))))

                if "batch" in locals():
                    batch.clear()
                    del batch
                if "num_infinite_params" in locals():
                    del num_infinite_params
                if "num_infinite_grad" in locals():
                    del num_infinite_grad
                if "model_outputs" in locals():
                    model_outputs.clear()
                    del model_outputs
                if "losses" in locals():
                    losses.clear()
                    del losses
                if "loss" in locals():
                    del loss

                gc.collect()
                torch.cuda.empty_cache()

                # simulate fake backward to keep processes in sync
                print("Dummy backprop on GPU %d, %r" % (comm.get_rank(), device))
                loss = 0.0 * sum(p.sum() for p in params)
                loss.backward()


            # When training with normal loss, sometimes I get NaNs in gradient that
            # cause the model to explode. Check for this before performing a gradient
            # update. This is safe in mult-GPU since gradients have already been
            # summed, so each GPU has the same gradients.
            num_infinite_grad = 0
            for p in params:
                if p.grad is not None:
                    num_infinite_grad += (torch.isfinite(p.grad) == 0).sum() \
                                                                      .item()
            if num_infinite_grad == 0:
                optimizer.step()
            else:
                msg = "WARNING: Got %d non-finite elements in gradient; skipping update"
                logger.info(msg % num_infinite_grad)

            # clean cuda cache to save memory
            if torch.cuda.is_available() and cp.t % 1 == 0:
                if "batch" in locals():
                    batch.clear()
                    del batch
                if "num_infinite_params" in locals():
                    del num_infinite_params
                if "num_infinite_grad" in locals():
                    del num_infinite_grad
                if "model_outputs" in locals():
                    model_outputs.clear()
                    del model_outputs
                if "losses" in locals():
                    losses.clear()
                    del losses
                if "loss" in locals():
                    del loss

                gc.collect()
                # logger.info("clearing cuda cache")
                torch.cuda.empty_cache()

            cp.step()

        cp.step_epoch()
        eval_and_save(
            model, loaders, optimizer, scheduler, cp,
            cfg.SOLVER.EARLY_STOP_METRIC
        )

    if comm.is_main_process():
        logger.info("Evaluating on test set:")
        test_loader = build_data_loader(
            cfg, get_dataset_name(cfg), "test", multigpu=False
        )
        evaluate_test_p2m(model, test_loader)


def eval_and_save(
    model, loaders, optimizer, scheduler, cp, early_stop_metric
):
    # NOTE(gkioxari) For now only do evaluation on the main process
    if comm.is_main_process():
        logger.info("Evaluating on training set:")
        train_metrics, train_preds = evaluate_split(
            model, loaders["train_eval"], prefix="train_", max_predictions=1000
        )
        eval_split = "val"
        if eval_split not in loaders:
            logger.info("WARNING: No val set!!! Computing metrics on test set!")
            eval_split = "test"
        logger.info("Evaluating on %s set:" % eval_split)
        test_metrics, test_preds = evaluate_split(
            model, loaders[eval_split], prefix="%s_" % eval_split, max_predictions=1000
        )
        str_out = "Results on train: "
        for k, v in train_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)
        str_out = "Results on %s: " % eval_split
        for k, v in test_metrics.items():
            str_out += "%s %.4f " % (k, v)
        logger.info(str_out)

        # The main process is responsible for managing the checkpoint
        # TODO(gkioxari) revisit these stores
        """
        cp.store_metric(**train_preds)
        cp.store_metric(**test_preds)
        """
        cp.store_metric(**train_metrics)
        cp.store_metric(**test_metrics)
        cp.early_stop_metric = eval_split + "_" + early_stop_metric

        cp.store_state("model", model.state_dict())
        cp.store_state("optim", optimizer.state_dict())
        cp.store_state("lr_scheduler", scheduler.state_dict())
        cp.save()

    # Since evaluation and checkpointing only happens on the main process,
    # make all processes wait
    if comm.get_world_size() > 1:
        dist.barrier()


@torch.no_grad()
def save_predictions(
    model, loader, output_dir,
    save_point_clouds, save_meshes, save_initial_meshes
):
    """
    This function is used save predicted and gt meshes
    """
    # Note that all eval runs on main process
    assert comm.is_main_process()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    device = torch.device("cuda:0")
    for i in range(3):
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

    # for loader_idx, batch in enumerate(tqdm.tqdm(loader)):
    for loader_idx, batch in enumerate(loader):
        batch = loader.postprocess(batch, device)

        model_kwargs = {}
        module = model.module if hasattr(model, "module") else model
        if isinstance(module, VoxMeshMultiViewHead):
            model_kwargs["intrinsics"] = batch["intrinsics"]
            model_kwargs["extrinsics"] = batch["extrinsics"]
        if isinstance(module, VoxMeshDepthHead):
            model_kwargs["masks"] = batch["masks"]
            if module.mvsnet is None:
                model_kwargs["depths"] = batch["depths"]
        model_outputs = model(batch["imgs"], **model_kwargs)

        # TODO: debug only
        # save_debug_predictions(batch, model_outputs)
        # continue
        # if loader_idx > 2:
        #     break

        gt_mesh = batch["meshes"]
        gt_mesh = gt_mesh.scale_verts(P2M_SCALE)
        gt_points = sample_points_from_meshes(
            gt_mesh, NUM_GT_SURFACE_SAMPLES, return_normals=False
        )
        gt_points = gt_points.cpu().detach().numpy()

        # save depth clouds
        # for batch_idx, views in enumerate(model_outputs.get("depth_clouds", [])):
        #     label, label_appendix = batch["id_strs"][batch_idx].split("-")[:2]
        #     for view_idx, view in enumerate(views):
        #         filename = os.path.join(
        #             output_dir,
        #             "{}_{}_{}_depthcloud.xyz" \
        #                 .format(label, label_appendix, view_idx)
        #         )
        #         np.savetxt(
        #             filename, (view * P2M_SCALE).cpu().detach().numpy()
        #         )

        if save_initial_meshes and "init_meshes" in model_outputs:
            save_p2m_format(
                batch, model_outputs["init_meshes"],
                gt_mesh, gt_points, output_dir, "10",
                False, True
            )


        if "meshes_pred" in model_outputs:
            # only the last stage
            # gcn_stages = [len(model_outputs["meshes_pred"])-1]
            # all stages
            gcn_stages = range(len(model_outputs["meshes_pred"]))
            for gcn_stage in gcn_stages:
                pred_mesh = model_outputs["meshes_pred"][gcn_stage]
                file_prefix = str(gcn_stage)
                save_p2m_format(
                    batch, pred_mesh, gt_mesh, gt_points, output_dir, file_prefix,
                    save_point_clouds, save_meshes
                )

def save_p2m_format(
        batch, pred_mesh, gt_mesh, gt_points, output_dir, file_prefix,
        save_point_clouds, save_meshes
):
    pred_mesh = pred_mesh.scale_verts(P2M_SCALE)

    if save_point_clouds:
        pred_points = sample_points_from_meshes(
            pred_mesh, NUM_PRED_SURFACE_SAMPLES, return_normals=False
        )
        pred_points = pred_points.cpu().detach().numpy()

    batch_size = gt_points.shape[0]
    for batch_idx in range(batch_size):
        label, label_appendix = batch["id_strs"][batch_idx].split("-")[:2]
        os.makedirs(os.path.join(output_dir, file_prefix), exist_ok=True)

        pred_filename = os.path.join(
            os.path.join(output_dir, file_prefix),
            "{}_{}_predict.xyz".format(label, label_appendix)
        )

        gt_filename = os.path.join(
            os.path.join(output_dir, file_prefix),
            "{}_{}_ground.xyz".format(label, label_appendix)
        )

        if save_point_clouds:
            np.savetxt(pred_filename, pred_points[batch_idx])
            np.savetxt(gt_filename, gt_points[batch_idx])

        if save_meshes:
            pred_filename = pred_filename.replace(".xyz", ".obj")
            gt_filename = gt_filename.replace(".xyz", ".obj")

            pred_verts, pred_faces = pred_mesh[batch_idx] \
                                        .get_mesh_verts_faces(0)
            gt_verts, gt_faces = gt_mesh[batch_idx] \
                                    .get_mesh_verts_faces(0)
            save_obj(pred_filename, pred_verts, pred_faces)
            # save_obj(gt_filename, gt_verts, gt_faces)

            # metrics = compare_meshes(
            #     pred_mesh[batch_idx], gt_mesh[batch_idx],
            #     num_samples=NUM_GT_SURFACE_SAMPLES, scale=1.0,
            #     thresholds=[0.01, 0.014142], reduce=True
            # )
            # print("%s_%s: %r" % (label, label_appendix, metrics))


@torch.no_grad()
def save_debug_predictions(batch, model_outputs):
    """
    save voxels and depths
    """
    from shapenet.modeling.voxel_ops import cubify
    from shapenet.modeling.mesh_arch import save_depths

    def save_meshes(meshes, id_strs, file_suffix):
        for batch_idx in range(len(meshes)):
            label, label_appendix = id_strs[batch_idx].split("-")[:2]
            save_obj(
                "/tmp/{}_{}_{}_{}.obj".format(
                    label, label_appendix, view_idx, file_suffix
                ),
                meshes[batch_idx].verts_packed(),
                meshes[batch_idx].faces_packed()
            )


    def save_cubified_voxels(voxels, id_strs, file_suffix):
        cubified = cubify(voxels, 48, 0.2)
        save_meshes(cubified, id_strs, file_suffix)


    batch_size = len(batch["id_strs"])
    if model_outputs.get("voxel_scores", None) is not None:
        for view_idx, voxels in enumerate(model_outputs["voxel_scores"]):
            save_cubified_voxels(
                voxels, batch["id_strs"], "{}_multiview_vox".format(view_idx)
            )

        save_cubified_voxels(
            model_outputs["merged_voxel_scores"], batch["id_strs"], "merged_vox"
        )
        save_cubified_voxels(
            model_outputs["merged_voxel_scores_old"], batch["id_strs"], "merged_vox_old"
        )

    if model_outputs.get("init_meshes", None) is not None:
        save_meshes(model_outputs["init_meshes"], batch["id_strs"])

    for stage_idx, pred_mesh in enumerate(model_outputs["meshes_pred"]):
        for batch_idx in range(batch_size):
            label, label_appendix = batch["id_strs"][batch_idx].split("-")[:2]
            # dim: num_points x 1 x num_views
            view_weights = model_outputs["view_weights"][stage_idx][batch_idx]
            # dim: num_points x num_views
            view_weights = F.normalize(view_weights, dim=-1).squeeze(1)
            view_weights = view_weights.detach().cpu().numpy()
            # show only the view with maximum weight
            max_indices = np.argmax(view_weights, axis=-1)
            # one hot encoded indices
            one_hot = np.eye(view_weights.shape[1])[max_indices]
            max_view_weights = view_weights * one_hot

            save_obj("/tmp/{}_{}_{}_pred_mesh.obj".format(
                label, label_appendix, stage_idx
            ), pred_mesh[0].verts_packed(), pred_mesh[0].faces_packed())

            num_views = model_outputs["view_weights"][stage_idx].shape[-1]
            global_pred_points = pred_mesh[0].verts_packed()
            global_gt_points = batch["points"][0]
            points_color = o3d.utility.Vector3dVector(max_view_weights)
            rel_extrinsics  = relative_extrinsics(batch["extrinsics"], batch["extrinsics"][:, 0])

            # visualize weights for each view
            for view_idx in range(num_views):
                transform = Transform3d(
                    matrix=rel_extrinsics[batch_idx, view_idx].transpose(0, 1),
                    dtype=global_pred_points.dtype, device=global_pred_points.device
                )

                # pred points
                local_pred_points = transform.transform_points(global_pred_points)
                point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                    local_pred_points.cpu().numpy()
                ))
                point_cloud.colors = points_color
                o3d.io.write_point_cloud("/tmp/{}_{}_{}_{}_pred_cloud.ply".format(
                    label, label_appendix, stage_idx, view_idx
                ), point_cloud)

                # gt points
                local_gt_points = transform.transform_points(global_gt_points)
                point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                    local_gt_points.cpu().numpy()
                ))
                # gt weights for ground truth corresponding to view idx
                gt_view_weights = np.zeros(
                    (local_gt_points.shape[0], num_views), dtype=np.float64
                )
                gt_view_weights[:, view_idx] = 1.0
                point_cloud.colors = o3d.utility.Vector3dVector(gt_view_weights)
                o3d.io.write_point_cloud("/tmp/{}_{}_{}_{}_gt_cloud.ply".format(
                    label, label_appendix, stage_idx, view_idx
                ), point_cloud)

    if "pred_depths" in model_outputs:
        masks = F.interpolate(
            batch["masks"], model_outputs["pred_depths"].shape[-2:],
            mode="nearest"
        )
        masked_depths = model_outputs["pred_depths"] * masks
        # TODO: the labels won't be corrent when batch size > 1. Fix it
        save_depths(
            masked_depths,
            "pred_{}_{}".format(label, label_appendix), (137, 137)
        )
        save_backproj_depths(masked_depths, batch["id_strs"], "depth_cloud")

    if "rendered_depths" in model_outputs:
        # TODO: the labels won't be corrent when batch size > 1. Fix it
        for stage_idx, depth in enumerate(model_outputs["rendered_depths"]):
            save_depths(
                depth,
                "rendered_{}_{}_{}".format(label, label_appendix, stage_idx),
                (137, 137)
            )
            save_backproj_depths(
                depth, batch["id_strs"], "rendered_cloud_{}".format(stage_idx)
            )


@torch.no_grad()
def save_backproj_depths(depths, id_strs, prefix):
    depths = F.interpolate(depths, (224, 224))
    dtype = depths.dtype
    device = depths.device
    intrinsics = get_blender_intrinsic_matrix().type(dtype).to(device)
    depth_points = get_points_from_depths(depths, intrinsics)

    for batch_idx in range(len(depth_points)):
        label, label_appendix = id_strs[batch_idx].split("-")[:2]
        for view_idx in range(len(depth_points[batch_idx])):
            points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                depth_points[batch_idx][view_idx].detach().cpu().numpy()
            ))
            o3d.io.write_point_cloud("/tmp/{}_{}_{}_{}.ply".format(
                prefix, label, label_appendix, view_idx
            ), points)


def setup_loaders(cfg):
    loaders = {}
    loaders["train"] = build_data_loader(
        cfg, get_dataset_name(cfg), "train", multigpu=comm.get_world_size() > 1
    )

    # Since sampling the mesh is now coupled with the data loader, we need to
    # make two different Dataset / DataLoaders for the training set: one for
    # training which uses precomputd samples, and one for evaluation which uses
    # more samples and computes them on the fly. This is sort of gross.
    loaders["train_eval"] = build_data_loader(
        cfg, get_dataset_name(cfg), "train_eval", multigpu=False
    )

    loaders["val"] = build_data_loader(
        cfg, get_dataset_name(cfg), "val", multigpu=False
    )
    return loaders


def setup(args):
    """
    Create configs and setup logger from arguments and the given config file.
    """
    cfg = get_shapenet_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # register dataset
    data_dir = register_shapenet(cfg.DATASETS.NAME)
    cfg.DATASETS.DATA_DIR = data_dir
    # if data was copied the data dir has changed
    if args.copy_data:
        cfg.DATASETS.DATA_DIR = args.data_dir
    cfg.freeze()

    colorful_logging = not args.no_color
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    comm.synchronize()

    logger = setup_logger(
        output_dir, color=colorful_logging, name="shapenet", distributed_rank=comm.get_rank()
    )
    logger.info(
        "Using {} GPUs per machine. Rank of current process: {}".format(
            args.num_gpus, comm.get_rank()
        )
    )
    logger.info(args)

    logger.info("Environment info:\n" + collect_env_info())
    logger.info(
        "Loaded config file {}:\n{}".format(args.config_file, open(args.config_file, "r").read())
    )
    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        path = os.path.join(output_dir, "config.yaml")
        with open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))
    return cfg


def shapenet_launch():
    args = default_argument_parser()

    # Note we need this only for pretrained models with torchvision.
    os.environ["TORCH_HOME"] = args.torch_home

    if args.copy_data:
        # if copy data is 1 then you need to provide args.data_dir
        # from which to copy data
        if args.data_dir == "":
            raise ValueError("You need to provide args.data_dir")
        copy_data(args)

    if args.eval_only:
        main_worker_eval(0, args)
        return

    if args.num_gpus > 1:
        mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,), daemon=False)
    else:
        main_worker(0, args)


if __name__ == "__main__":
    shapenet_launch()
