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
from vilbert.datasets import VisualEntailmentDataset
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import pdb

logger = logging.getLogger(__name__)


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

    task_datasets_train = None
    if "train" in split:
        task_datasets_train = VisualEntailmentDataset(
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

    task_datasets_val = None
    if "val" in split:
        task_datasets_val = DatasetMapTrain[task_name](
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
            train_sampler = RandomSampler(task_datasets_train)
        else:
            # TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(task_datasets_train)

        task_dataloader_train = DataLoader(
            task_datasets_train,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters = len(task_dataloader_train)
        task_batch_size = batch_size

    if "val" in split:
        task_dataloader_val = DataLoader(
            task_datasets_val,
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