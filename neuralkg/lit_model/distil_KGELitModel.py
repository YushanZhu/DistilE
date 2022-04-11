from logging import debug
from pickletools import optimize
from tkinter.messagebox import NO
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from collections import defaultdict as ddict
from .BaseLitModel import BaseLitModel
from IPython import embed
from neuralkg.eval_task import *
from IPython import embed
from neuralkg import loss
from functools import partial

class distil_KGELitModel(BaseLitModel):
    """Processing of training, evaluation and testing.
    """

    def __init__(self, model, args = None, model_tea=None):
        super().__init__(model, args, )
        self.model = model
        self.model_tea = model_tea
        self.args = args
        optim_name = args.optim_name
        self.optimizer_class = getattr(torch.optim, optim_name)
        loss_name = args.loss_name
        self.loss_class = getattr(loss, loss_name)
        self.loss = self.loss_class(args, model, model_tea)

        # import pdb; pdb.set_trace()

    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lr", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        return parser
    
    def training_step(self, batch, batch_idx):
        """Getting samples and training in KG model.
        
        Args:
            batch: The training data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            loss: The training loss for back propagation.
        """
        pos_sample = batch["positive_sample"]
        neg_sample = batch["negative_sample"]
        mode = batch["mode"]
        pos_score = self.model(pos_sample) # [512, 1]
        neg_score = self.model(pos_sample, neg_sample, mode) # [512, 1024]

        # 计算teacher的score
        pos_score_tea = self.model_tea(pos_sample)
        neg_score_tea = self.model_tea(pos_sample, neg_sample, mode)

        # 取出对应的h和t embedding
        pos_head_emb = self.model.ent_emb(pos_sample[:, 0])  # [bs, 1, dim]
        pos_tail_emb = self.model.ent_emb(pos_sample[:, 2])  # [bs, 1, dim]

        pos_head_emb_tea = self.model_tea.ent_emb(pos_sample[:, 0])  # [bs, 1, dim]
        pos_tail_emb_tea = self.model_tea.ent_emb(pos_sample[:, 2])  # [bs, 1, dim]

        if self.args.use_weight:
            subsampling_weight = batch["subsampling_weight"]
            loss = self.loss(pos_score, neg_score, pos_score_tea, neg_score_tea, pos_head_emb, pos_tail_emb, pos_head_emb_tea, pos_tail_emb_tea, subsampling_weight)
        else:
            loss = self.loss(pos_score, neg_score, pos_score_tea, neg_score_tea, pos_head_emb, pos_tail_emb, pos_head_emb_tea, pos_tail_emb_tea, )
        self.log("Train|loss", loss,  on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Getting samples and validation in KG model.
        
        Args:
            batch: The evalutaion data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,3,10.
        """
        results = dict()
        ranks = link_predict(batch, self.model, prediction='all')
        ranks_tea = link_predict(batch, self.model_tea, prediction='all')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        results['t_mrr'] = torch.sum(1.0/ ranks_tea).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
            results['t_hits@{}'.format(k)] = torch.numel(ranks_tea[ranks_tea <= k])
        return results
    
    def validation_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Eval")
        # self.log("Eval|mrr", outputs["Eval|mrr"], on_epoch=True)
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
        print(outputs)

    def test_step(self, batch, batch_idx):
        """Getting samples and test in KG model.
        
        Args:
            batch: The evaluation data.
            batch_idx: The dict_key in batch, type: list.

        Returns:
            results: mrr and hits@1,3,10.
        """
        results = dict()
        ranks = link_predict(batch, self.model, prediction='all')
        results["count"] = torch.numel(ranks)
        results["mrr"] = torch.sum(1.0 / ranks).item()
        for k in self.args.calc_hits:
            results['hits@{}'.format(k)] = torch.numel(ranks[ranks <= k])
        return results
    
    def test_epoch_end(self, results) -> None:
        outputs = self.get_results(results, "Test")
        self.log_dict(outputs, prog_bar=True, on_epoch=True)
     

    def configure_optimizers(self):
        """Setting optimizer and lr_scheduler.

        Returns:
            optim_dict: Record the optimizer and lr_scheduler, type: dict.   
        """
        milestones = int(self.args.max_epochs / 2)
        #optimizer = self.optimizer_class(self.model.parameters(), lr=self.args.lr)
        optimizer = self.optimizer_class(self.loss.parameters(), lr = self.args.lr) # loss里包含了model  model_tea 以及sem全部参数
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestones], gamma=0.1)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': StepLR}
        return optim_dict
