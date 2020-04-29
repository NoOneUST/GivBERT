# MODIFIED FROM: https://github.com/facebookresearch/vilbert-multi-task
from io import open
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_transformers.tokenization_bert import BertTokenizer
from .datasets import VisualEntailmentDataset
from .datasets._image_features_reader import ImageFeaturesH5Reader
import pdb

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}

def LoadDatasets(args, task_cfg, split="trainval"):

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}

    if task_cfg["features_h5path1"] not in task_feature_reader1:
        task_feature_reader1[task_cfg["features_h5path1"]] = None
    if task_cfg["features_h5path2"] not in task_feature_reader2:
        task_feature_reader2[task_cfg["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )
    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_dataset_train = None
    task_dataset_val = None
    task_dataloader_train = None
    task_dataloader_val = None
    task_batch_size = 0
    task_num_iters = 0

    task_name = task_cfg["name"]
    batch_size = task_cfg["batch_size"] // args.gradient_accumulation_steps
    num_workers = args.num_workers
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())

    # num_workers = int(num_workers / len(ids))
    logger.info(
        "Loading %s Dataset with batch size %d"
        % (task_cfg["name"], batch_size)
    )

    if "train" in split:
        task_dataset_train = VisualEntailmentDataset(
            task=task_cfg["name"],
            dataroot=task_cfg["dataroot"],
            annotations_jsonpath=task_cfg["train_annotations_jsonpath"],
            split=task_cfg["train_split"],
            image_features_reader=task_feature_reader1[
                task_cfg["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=task_cfg["max_seq_length"],
            max_region_num=task_cfg["max_region_num"],
        )

    if "val" in split:
        task_dataset_val = VisualEntailmentDataset(
            task=task_cfg["name"],
            dataroot=task_cfg["dataroot"],
            annotations_jsonpath=task_cfg["val_annotations_jsonpath"],
            split=task_cfg["val_split"],
            image_features_reader=task_feature_reader1[
                task_cfg["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=task_cfg["max_seq_length"],
            max_region_num=task_cfg["max_region_num"],
        )


    if "train" in split:
        if args.local_rank == -1:
            train_sampler = RandomSampler(task_dataset_train)
        else:
            # TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(task_dataset_train)

        task_dataloader_train = DataLoader(
            task_dataset_train,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters = len(task_dataloader_train)
        task_batch_size = batch_size

    if "val" in split:
        task_dataloader_val = DataLoader(
            task_dataset_val,
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
        )

    return (
        task_batch_size,
        task_num_iters,
        task_dataset_train,
        task_dataset_val,
        task_dataloader_train,
        task_dataloader_val,
    )

def LoadDatasetEval(args, task_cfg):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}

    if task_cfg["features_h5path1"] not in task_feature_reader1:
        task_feature_reader1[task_cfg["features_h5path1"]] = None
    if task_cfg["features_h5path2"] not in task_feature_reader2:
        task_feature_reader2[task_cfg["features_h5path2"]] = None

    # initilzie the feature reader
    for features_h5path in task_feature_reader1.keys():
        if features_h5path != "":
            task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    for features_h5path in task_feature_reader2.keys():
        if features_h5path != "":
            task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
                features_h5path, args.in_memory
            )

    task_dataset_val = None
    task_dataloader_val = None
    task_batch_size = 0
    task_num_iters = 0

    task_name = task_cfg["name"]
    batch_size = args.batch_size
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    num_workers = int(args.num_workers / len(ids))
    logger.info(
        "Loading %s Dataset with batch size %d"
        % (task_cfg["name"], batch_size)
    )

    if args.split:
        eval_split = args.split
    else:
        eval_split = task_cfg["val_split"]

    task_dataset_val = VisualEntailmentDataset(
        task=task_cfg["name"],
        dataroot=task_cfg["dataroot"],
        annotations_jsonpath=task_cfg["val_annotations_jsonpath"],
        split=eval_split,
        image_features_reader=task_feature_reader1[
            task_cfg["features_h5path1"]
        ],
        gt_image_features_reader=task_feature_reader2[
            task_cfg["features_h5path2"]
        ],
        tokenizer=tokenizer,
        bert_model=args.bert_model,
        clean_datasets=args.clean_train_sets,
        padding_index=0,
        max_seq_length=task_cfg["max_seq_length"],
        max_region_num=task_cfg["max_region_num"],
    )

    task_dataloader_val = DataLoader(
        task_datasets_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
    )

    task_num_iters = len(task_dataloader_val)
    task_batch_size = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_dataset_val,
        task_dataloader_val,
    )


def ForwardModelsTrain(
    args,
    task_cfg,
    device,
    task_count,
    task_iter_train,
    task_dataloader_train,
    model,
    task_losses,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count % len(task_dataloader_train) == 0:
        task_iter_train = iter(task_dataloader_train)

    task_count += 1
    # get the batch
    batch = task_iter_train.next()
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
        batch
    )

    batch_size = features.size(0)

    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
    )

    loss = task_losses(vil_tri_prediction, target)
    loss = loss.mean()
    batch_score = compute_score_with_logits(
        vil_tri_prediction, target
    ).sum() / float(batch_size)

    return loss, batch_score

def ForwardModelsVal(args, task_cfg, device, batch, model, task_losses):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
        batch
    )

    batch_size = features.size(0)

    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
        question,
        features,
        spatials,
        segment_ids,
        input_mask,
        image_mask,
        co_attention_mask,
    )

    loss = task_losses(vil_tri_prediction, target)
    loss = loss.mean()
    batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

    return float(loss), float(batch_score), batch_size


def LoadLosses(args, task_cfg):
    losses = LossMap[task_cfg["loss"]]
    return losses

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def EvaluatingModel(
    args,
    task_cfg,
    device,
    batch,
    model,
    task_dataloader,
    task_loss,
    results,
    others,
):
    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
        batch
    )
    batch_size = features.size(0)


    with torch.no_grad():
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            image_mask,
            co_attention_mask,
        )

    loss = task_loss(vil_tri_prediction, target)
    loss = loss.mean()
    batch_score = compute_score_with_logits(vil_tri_prediction, target).sum()

    return float(loss), float(batch_score), batch_size, results, others