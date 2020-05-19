'''
SNLI-VE Parser
This file provide a sample code for parsing SNLI-VE dataset
Author: Ning Xie, xie.25@wright.edu
# Copyright (C) 2018 NEC Laboratories America, Inc. ("NECLA"). 
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
'''

import os
import jsonlines
import pickle
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import BertTokenizer, BertModel

from src.data_utils.utils import save, load, pad_sents, get_mask

Label2Number = {
    "contradiction": 0,
    "neutral":       1,
    "entailment":    2,
}


def parser(SNLI_VE_root, SNLI_VE_files, choice):
    '''
    This is a sample function to parse SNLI-VE dataset
    :param SNLI_VE_root: root of SNLI-VE dataset
    :param SNLI_VE_files: filenames of each data split of SNLI-VE
    :param choice: data split choice, train/dev/test
    '''
    filename = os.path.join(SNLI_VE_root, SNLI_VE_files[choice])
    image_ids = []
    labels = []
    hyps = []
    with jsonlines.open(filename) as jsonl_file:
        for line in jsonl_file:
            # #######################################################################
            # ############ Items used in our Visual Entailment (VE) Task ############
            # #######################################################################

            # => Flikr30kID can be used to find corresponding Flickr30k image premise
            Flikr30kID = str(line['Flikr30kID'])
            # =>  gold_label is the label assigned by the majority label in annotator_labels (at least 3 out of 5),
            # If such a consensus is not reached, the gold label is marked as "-",
            # which are already filtered out from our SNLI-VE dataset
            gold_label = str(line['gold_label'])
            # => hypothesis is the text hypothesis
            hypothesis = str(line['sentence2'])


            # # #######################################################################
            # # ######## Extra information for Possible Extensions of VE Task #########
            # # #######################################################################

            # # => hypothesis_binary_parse is the original hypothesis_binary_parse from SNLI dataset
            # hypothesis_binary_parse = str(line['sentence2_binary_parse'])
            # # => hypothesis_parse is the original hypothesis_parse from SNLI dataset
            # hypothesis_parse = str(line['sentence2_parse'])
            # # =>  annotator_labels is a list of annotations for current (premise, hypothesis) pair
            # annotator_labels = [str(item) for item in line['annotator_labels']]
            # # =>  captionID is the original image caption ID from SNLI dataset
            # captionID = str(line['captionID'])
            # # =>  pairID is the original (premise, hypothesis) pair ID from SNLI dataset
            # pairID = str(line['pairID'])
            # # => premise is the original text premise, which is not used in our VE task
            # premise = str(line['sentence1'])
            # # => premise_binary_parse is the original premise_binary_parse from SNLI dataset
            # premise_binary_parse = str(line['sentence1_binary_parse'])
            # # => premise_parse is the original premise_parse from SNLI dataset
            # premise_parse = str(line['sentence1_parse'])

            image_ids.append(Flickr30kID)
            labels.append(gold_label)
            hyps.append(hypothesis)
    return (image_ids, labels, hyps)

class SnliDataset(Dataset):
    def __init__(self, image_ids, labels, hyps, tokenizer):
        super(SnliDataset).__init__()
        token_ids = [tokenizer.encode(text=lines[i], add_special_tokens=True) for i in range(nums)]
        lens = get_lens(token_ids)
        mask = np.array(get_mask(token_ids))
        token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

        self.inputs = []
        self.mask = []
        self.labels = []
        self.id = []
        self.image = []
        self.lens = len(image_ids)
        self.tokenizer = tokenizer

        for num, (_id, label, hyps) in tqdm(enumerate(zip(image_ids, labels, hyps)),total=self.lens, desc='Initialize dataset'):
            self.input.append(self.tokenize(hyps))
            self.labels.append(Label2Number[label])
            self.image.append(_id)
            self.id.append(num)

        self.mask = get_mask(self.inputs)
        self.inputs = pad_sents(self.inputs)

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, index):
        return self.id[index], torch.LongTensor(self.inputs[index]), torch.LongTensor(self.mask[index]), self.image[index], self.labels[index]
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

def getDataloader(data, batch_size, test=False):
    shuffle = False if test else True

    image_ids, labels, hyps = data

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = SnliDataset(image_ids, labels, hyps, tokenizer)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle)
    return dataloader


def extractSnliFeatures(datafolder, choice, batch_size, test=False):
    # SNLI-VE paths
    SNLI_VE_root = datafolder
    SNLI_VE_files = {'dev': 'snli_ve_dev.jsonl',
                     'test': 'snli_ve_test.jsonl',
                     'train': 'snli_ve_train.jsonl'}

    data = parser(SNLI_VE_root, SNLI_VE_files, choice)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model = model.cuda()

    # get the dataloder
    dataloader = getDataloader(data, batch_size, test=False)
    # put the data samples into BertModel and get the output
    features = None
    for _id, _input, _mask, image_id, label in tqdm(dataloader, desc='Generate embeddings'):
        _input = _input.cuda()
        _mask = _mask.cuda

        feature = model(_input, _mask) # (batch, sen, emb_dim)
        feature = feature.cpu().numpy()

        if features is None:
            features = knowledge_embs
        else:
            features = np.concatenate((features, feature))
    # save the output of the BertModel
    # save({'ids': ids, 'embs': embs}, self.saving_path)

def LoadDatasets(args, task_cfg, ids, split="trainval"):

    if "roberta" in args.bert_model:
        tokenizer = RobertaTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    task = "TASK13"
    if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
        task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
    if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
        task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

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

    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    task = "TASK13"
    task_name = task_cfg[task]["name"]
    task_ids.append(task)
    batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
    num_workers = args.num_workers
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())
        num_workers = int(num_workers / dist.get_world_size())

    # num_workers = int(num_workers / len(ids))
    logger.info(
        "Loading %s Dataset with batch size %d"
        % (task_cfg[task]["name"], batch_size)
    )

    task_datasets_train[task] = None
    if "train" in split:
        task_datasets_train[task] = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
            split=task_cfg[task]["train_split"],
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

    task_datasets_val[task] = None
    if "val" in split:
        task_datasets_val[task] = DatasetMapTrain[task_name](
            task=task_cfg[task]["name"],
            dataroot=task_cfg[task]["dataroot"],
            annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
            split=task_cfg[task]["val_split"],
            image_features_reader=task_feature_reader1[
                task_cfg[task]["features_h5path1"]
            ],
            gt_image_features_reader=task_feature_reader2[
                task_cfg[task]["features_h5path2"]
            ],
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            clean_datasets=args.clean_train_sets,
            padding_index=0,
            max_seq_length=task_cfg[task]["max_seq_length"],
            max_region_num=task_cfg[task]["max_region_num"],
        )

    task_num_iters[task] = 0
    task_batch_sier=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_train[task])
        task_batch_size[task] = batch_size

    if "val" in split:
        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
        )O: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(task_datasets_train[task])

        task_dataloader_train[task] = DataLoader(
            task_datasets_train[task],
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        task_num_iters[task] = len(task_dataloader_train[task])
        task_batch_size[task] = batch_size

    if "val" in split:
        task_dataloader_val[task] = DataLoader(
            task_datasets_val[task],
            shuffle=False,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
        )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def LoadDatasetEval(args, task_cfg, ids):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    task = "TASK13"
    if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
        task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
    if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
        task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

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

    task_datasets_val = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}

    task_id= "13"
    task = "TASK" + task_id
    task_ids.append(task)
    task_name = task_cfg[task]["name"]
    batch_size = args.batch_size
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    num_workers = int(args.num_workers / len(ids))
    logger.info(
        "Loading %s Dataset with batch size %d"
        % (task_cfg[task]["name"], batch_size)
    )

    if args.split:
        eval_split = args.split
    else:
        eval_split = task_cfg[task]["val_split"]

    task_datasets_val[task] = DatasetMapEval[task_name](
        task=task_cfg[task]["name"],
        dataroot=task_cfg[task]["dataroot"],
        annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
        split=eval_split,
        image_features_reader=task_feature_reader1[
            task_cfg[task]["features_h5path1"]
        ],
        gt_image_features_reader=task_feature_reader2[
            task_cfg[task]["features_h5path2"]
        ],
        tokenizer=tokenizer,
        bert_model=args.bert_model,
        clean_datasets=args.clean_train_sets,
        padding_index=0,
        max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"],
    )

    task_dataloader_val[task] = DataLoader(
        task_datasets_val[task],
        shuffle=False,
        batch_size=batch_size,
        num_workers=10,
        pin_memory=True,
    )

    task_num_iters[task] = len(task_dataloader_val[task])
    task_batch_size[task] = batch_size

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_val,
        task_dataloader_val,
    )

if __name__ == '__main__':

    # SNLI-VE paths
    SNLI_VE_root = './'
    SNLI_VE_files = {'dev': 'snli_ve_dev.jsonl',
                     'test': 'snli_ve_test.jsonl',
                     'train': 'snli_ve_train.jsonl'}
    choice = 'dev'

    parser(SNLI_VE_root, SNLI_VE_files, choice)