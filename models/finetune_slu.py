#!/usr/bin/env python3
# -*- coding: utf8 -*-

import torch
import torch.nn as nn
from models.modules.context_embedder_base import ContextEmbedderBase


class SlotClassifyLayer(nn.Module):
    def __init__(self, input_size, num_tags, label2id, label_pad='[PAD]'):
        super(SlotClassifyLayer, self).__init__()
        self.num_tags = num_tags
        self.label2id = label2id
        self.label_pad = label_pad
        self.hidden2tag = nn.Linear(in_features=input_size, out_features=num_tags)
        self.logsoftmax = nn.LogSoftmax(dim=2)

        self.criterion = nn.NLLLoss()

    def forward(self, x, y=None):
        """
        :param x: torch.Tensor (batch_size, seq_len, n_in)
        :param y: torch.Tensor (batch_size, seq_len)
        :return:
        """
        tag_scores = self.hidden2tag(x)
        if self.training:
            tag_scores = self.logsoftmax(tag_scores)

        if self.label2id[self.label_pad] == 0:
            _, tag_result = torch.max(tag_scores[:, :, 1:], 2)  # block <pad> label as predict output
            tag_result.add_(1)
        else:
            _, tag_result = torch.max(tag_scores, 2)  # give up to block <pad> label for efficiency

        if self.training and y is not None:  # [PAD] label is blocked in init
            return tag_result, self.criterion(tag_scores.view(-1, self.num_tags), y.view(-1))
        else:
            return tag_result, torch.FloatTensor([0.0])


class IntentClassifyLayer(nn.Module):
    def __init__(self, input_size, num_tags, label2id, label_pad='[PAD]', loss_fn='ce'):
        super(IntentClassifyLayer, self).__init__()
        self.num_tags = num_tags
        self.label2id = label2id
        self.label_pad = label_pad

        self.intent_dropout = nn.Dropout(0.1)
        self.intent_linear = nn.Linear(input_size, num_tags)
        if loss_fn == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError

    def forward(self, x, y=None):
        """
        :param x: torch.Tensor (batch_size, 1, n_in)
        :param y: torch.Tensor (batch_size, 1)
        :return:
        """
        # intent
        intent_logits = self.intent_dropout(x)
        intent_logits = self.intent_linear(intent_logits)

        if self.label2id[self.label_pad] == 0:
            _, tag_result = torch.max(intent_logits[:, :, 1:], 2)  # block <pad> label as predict output
            tag_result.add_(1)
        else:
            _, tag_result = torch.max(intent_logits, 2)  # give up to block <pad> label for efficiency

        if self.training and y is not None:  # [PAD] label is blocked in init
            return tag_result, self.criterion(intent_logits.view(-1, self.num_tags), y.view(-1))
        else:
            return tag_result, torch.FloatTensor([0.0])


class BertSLU(nn.Module):

    def __init__(self, opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: nn.Module,
                 slot_decoder: nn.Module,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(BertSLU, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        self.intent_decoder = intent_decoder
        self.slot_decoder = slot_decoder
        self.config = config
        self.emb_log = emb_log

        self.no_embedder_grad = opt.no_embedder_grad
        self.label_mask = None

        # id2label
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        # fine-tune layers
        self.ft_layers = list(range(12-opt.ft_layer_num, 12))

    def get_context_reps(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            use_cls: bool = False,
    ):
        if self.no_embedder_grad:
            self.context_embedder.eval()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = False
        else:
            self.context_embedder.train()  # to avoid the dropout effect of reps model
            # self.context_embedder.requires_grad = True
            for name, param in self.context_embedder.embedder.named_parameters():
                if 'pooler' in name:
                    param.requires_grad = True
                elif 'encoder' in name and int(name.split('.')[2]) in self.ft_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if self.opt.do_debug:
            for name, param in self.context_embedder.embedder.named_parameters():
                print('get_context_reps : {} - {}'.format(name, param.requires_grad))
        seq_test_reps, sent_test_reps = self.context_embedder(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, use_cls=use_cls
        )
        if self.no_embedder_grad:
            seq_test_reps = seq_test_reps.detach()  # detach the reps part from graph
            sent_test_reps = sent_test_reps.detach()  # detach the reps part from graph
        return seq_test_reps, sent_test_reps

    def forward(self,
                test_token_ids: torch.Tensor,
                test_segment_ids: torch.Tensor,
                test_nwp_index: torch.Tensor,
                test_input_mask: torch.Tensor,
                slot_test_output_mask: torch.Tensor,
                intent_test_output_mask: torch.Tensor,
                slot_test_target: torch.Tensor = None,
                intent_test_target: torch.Tensor = None,
                support_num: torch.Tensor = None):

        # get token reps for slot & sent reps for intent
        # the size of input data of get_context_reps is : (batch_size, support_size, sent_len)
        # so, expand as that
        token_reps, sent_reps = self.get_context_reps(test_token_ids, test_segment_ids,
                                                      test_nwp_index, test_input_mask,
                                                      self.opt.use_cls)
        if self.opt.use_cls:
            sent_reps = sent_reps.squeeze(1)

        if self.training:
            loss = 0.
            if self.opt.task in ['slot', 'slu']:
                slot_output, slot_loss = self.slot_decoder(token_reps, slot_test_target)
                loss += slot_loss
            if self.opt.task in ['intent', 'slu']:
                intent_output, intent_loss = self.intent_decoder(sent_reps, intent_test_target)
                loss += intent_loss
            return loss
        else:
            output = {}
            if self.opt.task in ['slot', 'slu']:
                slot_output, _ = self.slot_decoder(token_reps)
                output['slot'] = slot_output
            if self.opt.task in ['intent', 'slu']:
                intent_output, _ = self.intent_decoder(sent_reps)
                output['intent'] = intent_output
            return output

