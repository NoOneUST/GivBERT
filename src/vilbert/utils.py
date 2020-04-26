from io import open
import json
import logging
from functools import wraps
from hashlib import sha256
from pathlib import Path
import os
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from functools import partial, wraps

import boto3
import requests
from botocore.exceptions import ClientError
from tqdm import tqdm
from tensorboardX import SummaryWriter
from time import gmtime, strftime
from bisect import bisect
from torch import nn
import torch
from torch._six import inf

import pdb

PYTORCH_PRETRAINED_BERT_CACHE = Path(
    os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", Path.home() / ".pytorch_pretrained_bert")
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TaskStopOnPlateau(object):
    def __init__(
        self,
        mode="min",
        patience=10,
        continue_threshold=0.005,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.in_stop = False
        self.eps = eps
        self.last_epoch = -1
        self.continue_threshold = continue_threshold
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )
        self._init_continue_is_better(
            mode="min", threshold=continue_threshold, threshold_mode=threshold_mode
        )
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.in_stop = False

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.in_stop = True
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        # if the perforance is keep dropping, then start optimizing again.
        elif self.continue_is_better(current, self.best) and self.in_stop:
            self.in_stop = False
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        # if we lower the learning rate, then
        # call reset.

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == "min" and threshold_mode == "rel":
            rel_epsilon = 1.0 - threshold
            return a < best * rel_epsilon

        elif mode == "min" and threshold_mode == "abs":
            return a < best - threshold

        elif mode == "max" and threshold_mode == "rel":
            rel_epsilon = threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def _init_continue_is_better(self, mode, threshold, threshold_mode):

        self.continue_is_better = partial(self._cmp, mode, threshold_mode, threshold)


class tbLogger(object):
    def __init__(
        self,
        log_dir,
        txt_dir,
        task_num_iters,
        gradient_accumulation_steps,
        save_logger=True,
        txt_name="out.txt",
    ):
        logger.info("logging file at: " + log_dir)

        self.save_logger = save_logger
        self.log_dir = log_dir
        self.txt_dir = txt_dir
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=log_dir)

        self.txt_f = open(txt_dir + "/" + txt_name, "w")
        self.task_loss = 0
        self.task_loss_tmp = 0 
        self.task_score_tmp = 0
        self.task_norm_tmp = 0 
        self.task_step = 0 
        self.task_step_tmp = 0 
        self.task_num_iters = task_num_iters
        self.epochId = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_loss_val = 0
        self.task_score_val = 0
        self.task_step_val = 0
        self.task_iter_val = 0
        self.task_datasize_val = 0

        self.masked_t_loss = 0
        self.masked_v_loss = 0
        self.next_sentense_loss = 0

        self.masked_t_loss_val = 
        self.masked_v_loss_val = 0
        self.next_sentense_loss_val = 0

    def __getstate__(self):
        d = dict(self.__dict__)
        del d["logger"]
        del d["txt_f"]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=self.log_dir)

        self.txt_f = open(self.txt_dir + "/" + "out.txt", "a")

    def txt_close(self):
        self.txt_f.close()

    def linePlot(self, step, val, split, key, xlabel="None"):
        if self.save_logger:
            self.logger.add_scalar(split + "/" + key, val, step)

    def step_train(self, epochId, stepId, loss, score, norm, split):

        self.task_loss += loss
        self.task_loss_tmp += loss
        self.task_score_tmp += score
        self.task_norm_tmp += norm
        self.task_step += self.gradient_accumulation_steps
        self.task_step_tmp += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(stepId, loss, split, "loss")
        self.linePlot(stepId, score, split, "score")
        self.linePlot(stepId, norm, split, "norm")

    def step_val(self, epochId, loss, score, batch_size, split):
        self.task_loss_val += loss * batch_size
        self.task_score_val += score
        self.task_step_val += self.gradient_accumulation_steps
        self.task_datasize_val += batch_size


    def showLossValAll(self):
        progressInfo = "Eval Ep: %d " % self.epochId
        lossInfo = "Validation "
        val_scores = {}
        ave_loss = 0

        loss = self.task_loss_val / float(self.task_step_val)
        score = self.task_score_val / float(
            self.task_datasize_val
        )
        val_scores = score
        ave_loss += loss
        lossInfo += "loss %.3f score %.3f " % (
            loss,
            score * 100.0,
        )

        self.linePlot(
            self.epochId, loss, "val", "loss"
        )
        self.linePlot(
            self.epochId, score, "val", "score"
        )

        self.task_loss_val = 0
        self.task_score_val = 0
        self.task_datasize_val = 0
        self.task_step_val = 0

        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return val_scores

    def getValScore(self):
        return self.task_score_val / float(self.task_datasize_val)

    def showLossVal(self, task_stop_controller=None):
        progressInfo = "Eval task %s on iteration %d " % (self.task_step)
        lossInfo = "Validation "
        ave_loss = 0
        loss = self.task_loss_val / float(self.task_datasize_val)
        score = self.task_score_val / float(self.task_datasize_val)
        ave_loss += loss
        lossInfo += "loss %.3f score %.3f " % (
            loss,
            score * 100.0,
        )

        self.linePlot(
            self.task_step, loss, "val", "loss"
        )
        self.linePlot(
            self.task_step, score, "val", "score"
        )
        if task_stop_controller is not None:
            self.linePlot(
                self.task_step,
                task_stop_controller.in_stop,
                "val",
                "early_stop",
            )

        self.task_loss_val = 0
        self.task_score_val = 0
        self.task_datasize_val = 0
        self.task_step_val = 0
        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return score

    def showLossTrain(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ""
        if self.task_num_iters > 0:
            if self.task_step_tmp:
                lossInfo += (
                    "iter %d Ep: %.2f loss %.3f score %.3f lr %.6g "
                    % (
                        self.task_step,
                        self.task_step
                        / float(self.task_num_iters),
                        self.task_loss_tmp
                        / float(self.task_step_tmp),
                        self.task_score_tmp
                        / float(self.task_step_tmp),
                        self.task_norm_tmp
                        / float(self.task_step_tmp),
                    )
                )

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = 0
        self.task_loss_tmp = 0
        self.task_score_tmp = 0
        self.task_norm_tmp = 0