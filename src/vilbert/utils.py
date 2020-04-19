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
        task_names,
        task_ids,
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
        self.task_id2name = {
            ids: name.replace("+", "plus") for ids, name in zip(task_ids, task_names)
        }
        self.task_ids = task_ids
        self.task_loss = {task_id: 0 for task_id in task_ids}
        self.task_loss_tmp = {task_id: 0 for task_id in task_ids}
        self.task_score_tmp = {task_id: 0 for task_id in task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in task_ids}
        self.task_step = {task_id: 0 for task_id in task_ids}
        self.task_step_tmp = {task_id: 0 for task_id in task_ids}
        self.task_num_iters = task_num_iters
        self.epochId = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_loss_val = {task_id: 0 for task_id in task_ids}
        self.task_score_val = {task_id: 0 for task_id in task_ids}
        self.task_step_val = {task_id: 0 for task_id in task_ids}
        self.task_iter_val = {task_id: 0 for task_id in task_ids}
        self.task_datasize_val = {task_id: 0 for task_id in task_ids}

        self.masked_t_loss = {task_id: 0 for task_id in task_ids}
        self.masked_v_loss = {task_id: 0 for task_id in task_ids}
        self.next_sentense_loss = {task_id: 0 for task_id in task_ids}

        self.masked_t_loss_val = {task_id: 0 for task_id in task_ids}
        self.masked_v_loss_val = {task_id: 0 for task_id in task_ids}
        self.next_sentense_loss_val = {task_id: 0 for task_id in task_ids}

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

    def step_train(self, epochId, stepId, loss, score, norm, task_id, split):

        self.task_loss[task_id] += loss
        self.task_loss_tmp[task_id] += loss
        self.task_score_tmp[task_id] += score
        self.task_norm_tmp[task_id] += norm
        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(stepId, loss, split, self.task_id2name[task_id] + "_loss")
        self.linePlot(stepId, score, split, self.task_id2name[task_id] + "_score")
        self.linePlot(stepId, norm, split, self.task_id2name[task_id] + "_norm")

    def step_train_CC(
        self,
        epochId,
        stepId,
        masked_loss_t,
        masked_loss_v,
        next_sentence_loss,
        norm,
        task_id,
        split,
    ):

        self.masked_t_loss[task_id] += masked_loss_t
        self.masked_v_loss[task_id] += masked_loss_v
        self.next_sentense_loss[task_id] += next_sentence_loss
        self.task_norm_tmp[task_id] += norm

        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(
            stepId, masked_loss_t, split, self.task_id2name[task_id] + "_masked_loss_t"
        )
        self.linePlot(
            stepId, masked_loss_v, split, self.task_id2name[task_id] + "_masked_loss_v"
        )
        self.linePlot(
            stepId,
            next_sentence_loss,
            split,
            self.task_id2name[task_id] + "_next_sentence_loss",
        )

    def step_val(self, epochId, loss, score, task_id, batch_size, split):
        self.task_loss_val[task_id] += loss * batch_size
        self.task_score_val[task_id] += score
        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size

    def step_val_CC(
        self,
        epochId,
        masked_loss_t,
        masked_loss_v,
        next_sentence_loss,
        task_id,
        batch_size,
        split,
    ):

        self.masked_t_loss_val[task_id] += masked_loss_t
        self.masked_v_loss_val[task_id] += masked_loss_v
        self.next_sentense_loss_val[task_id] += next_sentence_loss

        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size

    def showLossValAll(self):
        progressInfo = "Eval Ep: %d " % self.epochId
        lossInfo = "Validation "
        val_scores = {}
        ave_loss = 0
        for task_id in self.task_ids:
            loss = self.task_loss_val[task_id] / float(self.task_step_val[task_id])
            score = self.task_score_val[task_id] / float(
                self.task_datasize_val[task_id]
            )
            val_scores[task_id] = score
            ave_loss += loss
            lossInfo += "[%s]: loss %.3f score %.3f " % (
                self.task_id2name[task_id],
                loss,
                score * 100.0,
            )

            self.linePlot(
                self.epochId, loss, "val", self.task_id2name[task_id] + "_loss"
            )
            self.linePlot(
                self.epochId, score, "val", self.task_id2name[task_id] + "_score"
            )

        self.task_loss_val = {task_id: 0 for task_id in self.task_loss_val}
        self.task_score_val = {task_id: 0 for task_id in self.task_score_val}
        self.task_datasize_val = {task_id: 0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id: 0 for task_id in self.task_ids}

        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return val_scores

    def getValScore(self, task_id):
        return self.task_score_val[task_id] / float(self.task_datasize_val[task_id])

    def showLossVal(self, task_id, task_stop_controller=None):
        progressInfo = "Eval task %s on iteration %d " % (
            task_id,
            self.task_step[task_id],
        )
        lossInfo = "Validation "
        ave_loss = 0
        loss = self.task_loss_val[task_id] / float(self.task_datasize_val[task_id])
        score = self.task_score_val[task_id] / float(self.task_datasize_val[task_id])
        ave_loss += loss
        lossInfo += "[%s]: loss %.3f score %.3f " % (
            self.task_id2name[task_id],
            loss,
            score * 100.0,
        )

        self.linePlot(
            self.task_step[task_id], loss, "val", self.task_id2name[task_id] + "_loss"
        )
        self.linePlot(
            self.task_step[task_id], score, "val", self.task_id2name[task_id] + "_score"
        )
        if task_stop_controller is not None:
            self.linePlot(
                self.task_step[task_id],
                task_stop_controller[task_id].in_stop,
                "val",
                self.task_id2name[task_id] + "_early_stop",
            )

        self.task_loss_val[task_id] = 0
        self.task_score_val[task_id] = 0
        self.task_datasize_val[task_id] = 0
        self.task_step_val[task_id] = 0
        logger.info(progressInfo)
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return score

    def showLossTrain(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ""
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += (
                        "[%s]: iter %d Ep: %.2f loss %.3f score %.3f lr %.6g "
                        % (
                            self.task_id2name[task_id],
                            self.task_step[task_id],
                            self.task_step[task_id]
                            / float(self.task_num_iters[task_id]),
                            self.task_loss_tmp[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.task_score_tmp[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.task_norm_tmp[task_id]
                            / float(self.task_step_tmp[task_id]),
                        )
                    )

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_loss_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_score_tmp = {task_id: 0 for task_id in self.task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in self.task_ids}

    def showLossValCC(self):
        progressInfo = "Eval Ep: %d " % self.epochId
        lossInfo = "Validation "
        for task_id in self.task_ids:
            masked_t_loss_val = self.masked_t_loss_val[task_id] / float(
                self.task_step_val[task_id]
            )
            masked_v_loss_val = self.masked_v_loss_val[task_id] / float(
                self.task_step_val[task_id]
            )
            next_sentense_loss_val = self.next_sentense_loss_val[task_id] / float(
                self.task_step_val[task_id]
            )

            lossInfo += "[%s]: masked_t %.3f masked_v %.3f NSP %.3f" % (
                self.task_id2name[task_id],
                masked_t_loss_val,
                masked_v_loss_val,
                next_sentense_loss_val,
            )

            self.linePlot(
                self.epochId,
                masked_t_loss_val,
                "val",
                self.task_id2name[task_id] + "_mask_t",
            )
            self.linePlot(
                self.epochId,
                masked_v_loss_val,
                "val",
                self.task_id2name[task_id] + "_maks_v",
            )
            self.linePlot(
                self.epochId,
                next_sentense_loss_val,
                "val",
                self.task_id2name[task_id] + "_nsp",
            )

        self.masked_t_loss_val = {task_id: 0 for task_id in self.masked_t_loss_val}
        self.masked_v_loss_val = {task_id: 0 for task_id in self.masked_v_loss_val}
        self.next_sentense_loss_val = {
            task_id: 0 for task_id in self.next_sentense_loss_val
        }
        self.task_datasize_val = {task_id: 0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id: 0 for task_id in self.task_ids}

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

    def showLossTrainCC(self):
        # show the current loss, once showed, reset the loss.
        lossInfo = ""
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += (
                        "[%s]: iter %d Ep: %.2f masked_t %.3f masked_v %.3f NSP %.3f lr %.6g"
                        % (
                            self.task_id2name[task_id],
                            self.task_step[task_id],
                            self.task_step[task_id]
                            / float(self.task_num_iters[task_id]),
                            self.masked_t_loss[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.masked_v_loss[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.next_sentense_loss[task_id]
                            / float(self.task_step_tmp[task_id]),
                            self.task_norm_tmp[task_id]
                            / float(self.task_step_tmp[task_id]),
                        )
                    )

        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id: 0 for task_id in self.task_ids}
        self.masked_t_loss = {task_id: 0 for task_id in self.task_ids}
        self.masked_v_loss = {task_id: 0 for task_id in self.task_ids}
        self.next_sentense_loss = {task_id: 0 for task_id in self.task_ids}
        self.task_norm_tmp = {task_id: 0 for task_id in self.task_ids}