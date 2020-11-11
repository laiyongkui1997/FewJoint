#!/usr/bin/env python
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Tuple, Dict, List
from models.modules.context_embedder_base import ContextEmbedderBase
from models.few_shot_text_classifier import FewShotTextClassifier
from models.few_shot_seq_labeler import FewShotSeqLabeler
from collections import Counter
from models.modules.similarity_scorer_base import reps_dot, reps_l2_sim, reps_cosine_sim
from models.hopfield import Hopfield, HopfieldCore, HopfieldPooling


class FewShotSLU(torch.nn.Module):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(FewShotSLU, self).__init__()
        self.opt = opt
        self.context_embedder = context_embedder
        self.intent_decoder = intent_decoder
        self.slot_decoder = slot_decoder
        self.config = config
        self.emb_log = emb_log

        self.no_embedder_grad = opt.no_embedder_grad
        self.label_mask = None

        # separate loss dict
        self.loss_dict = {}

        # learning task
        self.learning_task = self.opt.task

        # id2label
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

    def set_learning_task(self, task):
        if task in ['intent', 'slot_filling', 'slu']:
            self.learning_task = task
        else:
            raise TypeError('the task `{}` is not defined'.format(task))

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        # reps for tokens: (batch_size, support_size, nwp_sent_len, emb_len)
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' getting emission '''
        intent_emission, slot_emission = None, None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(sent_test_reps, intent_test_output_mask,
                                                               sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(seq_test_reps, slot_test_output_mask,
                                                           seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)

        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                intent_loss = self.intent_decoder(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                  intent_support_output_mask, intent_test_target,
                                                  intent_support_target, self.label_mask)
                if self.opt.do_debug:
                    print("intent_loss: {}".format(intent_loss.item()))
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss
            if self.learning_task in ['slot_filling', 'slu']:
                slot_loss = self.slot_decoder(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                              slot_support_output_mask, slot_test_target, slot_support_target,
                                              self.label_mask)
                if self.opt.do_debug:
                    print("slot_loss: {}".format(slot_loss.item()))
                loss += slot_loss
                self.loss_dict['slot'] = slot_loss
            return loss
        else:
            '''store visualization embedding'''
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}

    def get_loss_dict(self):
        return self.loss_dict

    def get_context_reps(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            use_cls: bool = False,
    ):
        if self.no_embedder_grad:
            self.context_embedder.eval()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = False
        else:
            self.context_embedder.train()  # to avoid the dropout effect of reps model
            self.context_embedder.requires_grad = True
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.context_embedder(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids,
            support_segment_ids,
            support_nwp_index, support_input_mask, use_cls=use_cls
        )
        if self.no_embedder_grad:
            seq_test_reps = seq_test_reps.detach()  # detach the reps part from graph
            seq_support_reps = seq_support_reps.detach()  # detach the reps part from graph
            sent_test_reps = sent_test_reps.detach()  # detach the reps part from graph
            sent_support_reps = sent_support_reps.detach()  # detach the reps part from graph
        return seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps

    def onehot2label_id(self, inputs):
        origin_size = list(inputs.size())
        num_tags = origin_size[-1]
        inputs = inputs.view(-1, num_tags)
        after_size = inputs.size()[0]
        outputs = []
        for a_idx in range(after_size):
            tags = inputs[a_idx].tolist()
            if sum(tags) > 0:
                label_id = tags.index(1)
            else:  # the padding
                label_id = 0
            outputs.append(label_id)
        origin_size[-1] = 1
        outputs = torch.tensor(outputs, dtype=torch.long)
        outputs = outputs.reshape(*origin_size).squeeze(-1)
        return outputs

    def expand_intent_emission_to_slot(self, slot_support_target, intent_support_target,
                                       slot_id2label, intent_id2label, intent_emission) -> torch.Tensor:
        """
        get intent emission reps by deriving from slot emission reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param intent_emission: (batch_size, test_len, no_pad_intent_num_tags)
        :return: (batch_size, test_len, no_pad_intent_num_tags)
        """
        slot_support_target = self.onehot2label_id(slot_support_target)
        intent_support_target = self.onehot2label_id(intent_support_target)
        # get the size info
        batch_size, support_size, test_len = slot_support_target.size()
        # the test_len == 1, so remove it
        intent_emission = intent_emission.view(batch_size, -1)

        intent_support_target = intent_support_target.view(batch_size, -1).tolist()

        batch_expand_intent_emission = []
        for b_idx in range(batch_size):
            slot2intent_lst = {slot_id: Counter() for slot_id in slot_id2label}
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx]
                for slot_label_id in slot_support_target[b_idx][s_idx]:
                    if slot_id2label[slot_label_id.item()] not in ['O', '[PAD]']:
                        slot2intent_lst[slot_label_id.item()][intent_label_id] += 1

            expand_intent_emission = []
            for slot_id in slot_id2label:
                if slot_id == 0:  # the [PAD] label is removed
                    continue
                if len(slot2intent_lst[slot_id]) == 0:
                    expand_intent_emission.append(torch.tensor(0, dtype=torch.float).to(intent_emission.device))
                else:
                    all_intent_label_ids = list(slot2intent_lst[slot_id].keys())
                    all_intent_label_ids = [label_id - 1 for label_id in all_intent_label_ids]  # remove [pad] label
                    mean_val = torch.mean(intent_emission[b_idx][all_intent_label_ids], dim=0)
                    expand_intent_emission.append(mean_val)
            expand_intent_emission = torch.stack(expand_intent_emission, dim=0)
            batch_expand_intent_emission.append(expand_intent_emission)

        batch_expand_intent_emission = torch.stack(batch_expand_intent_emission, dim=0)
        batch_expand_intent_emission = batch_expand_intent_emission.unsqueeze(1)  # add test_len == 1
        return batch_expand_intent_emission

    def expand_slot_emission_to_intent(self, slot_support_target, intent_support_target,
                                       slot_id2label, intent_id2label, slot_emission) -> torch.Tensor:
        """
        get intent emission reps by deriving from slot emission reps, via support set statistic
        :param slot_support_target: (batch_size, support_size, test_len)
        :param intent_support_target: (batch_size, support_size, 1)
        :param slot_id2label:
        :param intent_id2label:
        :param slot_emission: (batch_size, test_len, no_pad_intent_num_tags)
        :return: (batch_size, test_len, no_pad_intent_num_tags)
        """
        slot_support_target = self.onehot2label_id(slot_support_target)
        intent_support_target = self.onehot2label_id(intent_support_target)

        batch_size, support_size, test_len = slot_support_target.size()

        intent_support_target = intent_support_target.view(batch_size, -1).tolist()

        batch_expand_slot_emission = []
        for b_idx in range(batch_size):
            intent2slot_lst = {intent_id: Counter() for intent_id in intent_id2label}
            for s_idx in range(support_size):
                intent_label_id = intent_support_target[b_idx][s_idx]
                for slot_label_id in slot_support_target[b_idx][s_idx]:
                    if slot_id2label[slot_label_id.item()] not in ['O', '[PAD]']:
                        intent2slot_lst[intent_label_id][slot_label_id.item()] += 1

            expand_slot_emission = []
            for intent_id in intent_id2label:
                if intent_id == 0:  # the [PAD] label is removed
                    continue
                if len(intent2slot_lst[intent_id]) == 0:
                    expand_slot_emission.append(torch.tensor(0, dtype=torch.float).to(slot_emission.device))
                else:
                    all_slot_label_ids = list(intent2slot_lst[intent_id].keys())
                    all_slot_label_ids = [label_id - 1 for label_id in all_slot_label_ids]  # remove [pad] label
                    mean_val = torch.mean(slot_emission[b_idx][:, all_slot_label_ids])
                    expand_slot_emission.append(mean_val)
            expand_slot_emission = torch.stack(expand_slot_emission, dim=0)
            batch_expand_slot_emission.append(expand_slot_emission)

        batch_expand_slot_emission = torch.stack(batch_expand_slot_emission, dim=0)
        batch_expand_slot_emission = batch_expand_slot_emission.unsqueeze(1)
        return batch_expand_slot_emission


class EmissionMergeIterationFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None, ):
        super(EmissionMergeIterationFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                               config, emb_log)
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert']:
            emb_dim = 768
        else:
            emb_dim = opt.emb_dim

        # metric layer
        self.intent_metric = torch.nn.Linear(emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(emb_dim, opt.metric_dim)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        metric_sent_test_reps, metric_sent_support_reps = sent_test_reps, sent_support_reps  # intent
        metric_seq_test_reps, metric_seq_support_reps = seq_test_reps, seq_support_reps  # slot

        atv_lst = self.opt.metric_activation.split('-')
        intent_activation = atv_lst[0]
        slot_activation = atv_lst[0] if len(atv_lst) == 1 else atv_lst[1]

        if self.opt.split_metric in ['intent', 'both']:
            # intent metric space
            metric_sent_test_reps = self.intent_metric(metric_sent_test_reps)
            metric_sent_support_reps = self.intent_metric(metric_sent_support_reps)
            # activation
            if intent_activation == 'relu':
                metric_sent_test_reps = F.relu(metric_sent_test_reps)
                metric_sent_support_reps = F.relu(metric_sent_support_reps)
            elif intent_activation == 'sigmoid':
                metric_sent_test_reps = torch.sigmoid(metric_sent_test_reps)
                metric_sent_support_reps = torch.sigmoid(metric_sent_support_reps)
            elif intent_activation == 'tanh':
                metric_sent_test_reps = torch.tanh(metric_sent_test_reps)
                metric_sent_support_reps = torch.tanh(metric_sent_support_reps)
            elif intent_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        if self.opt.split_metric in ['slot', 'both']:
            # slot metric space
            metric_seq_test_reps = self.slot_metric(metric_seq_test_reps)
            metric_seq_support_reps = self.slot_metric(metric_seq_support_reps)
            # activation
            if slot_activation == 'relu':
                metric_seq_test_reps = F.relu(metric_seq_test_reps)
                metric_seq_support_reps = F.relu(metric_seq_support_reps)
            elif slot_activation == 'sigmoid':
                metric_seq_test_reps = torch.sigmoid(metric_seq_test_reps)
                metric_seq_support_reps = torch.sigmoid(metric_seq_support_reps)
            elif slot_activation == 'tanh':
                metric_seq_test_reps = torch.tanh(metric_seq_test_reps)
                metric_seq_support_reps = torch.tanh(metric_seq_support_reps)
            elif slot_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        ''' getting emission '''
        intent_emission = None
        slot_emission = None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(metric_sent_test_reps, intent_test_output_mask,
                                                               metric_sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(metric_seq_test_reps, slot_test_output_mask,
                                                           metric_seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)

        ''' emission merge '''
        if self.opt.emission_merge_iter_num > 0:  # merge iteration
            # check weather the task definition allow the operation
            if intent_emission is None or slot_emission is None:
                raise ValueError('single task can not use iterative emission merge methods')
            # start merge iteration
            for i in range(self.opt.emission_merge_iter_num):
                # expand emission as same as the one of the other task
                expand_intent_emission = self.expand_intent_emission_to_slot(slot_support_target, intent_support_target,
                                                                             self.slot_id2label, self.intent_id2label,
                                                                             intent_emission)
                expand_slot_emission = self.expand_slot_emission_to_intent(slot_support_target, intent_support_target,
                                                                           self.slot_id2label, self.intent_id2label,
                                                                           slot_emission)
                # update emission with the one of the other task
                # update method is to concat them
                self.intent_decoder.cat_emission(expand_slot_emission)
                self.slot_decoder.cat_emission(expand_intent_emission)
                # get the updated emission
                intent_emission = self.intent_decoder.get_emission()
                slot_emission = self.slot_decoder.get_emission()
        else:  # none merge
            pass

        ''' return the result (loss / prediction) '''
        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                intent_loss = self.intent_decoder(metric_sent_test_reps, intent_test_output_mask, metric_sent_support_reps,
                                                  intent_support_output_mask, intent_test_target,
                                                  intent_support_target, self.label_mask)
                if self.opt.do_debug:
                    print('intent loss: {}'.format(intent_loss))
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss

            if self.learning_task in ['slot_filling', 'slu']:
                slot_loss = self.slot_decoder(metric_seq_test_reps, slot_test_output_mask, metric_seq_support_reps,
                                              slot_support_output_mask, slot_test_target, slot_support_target,
                                              self.label_mask)
                if self.opt.do_debug:
                    print('slot loss: {}'.format(slot_loss))
                loss += slot_loss
                self.loss_dict['slot'] = slot_loss

            return loss
        else:
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(metric_sent_test_reps, intent_test_output_mask,
                                                          metric_sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(metric_seq_test_reps, slot_test_output_mask,
                                                      metric_seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask,)
            return {'slot': slot_preds, 'intent': intent_preds}


class EmissionMergeIntentFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None, ):
        super(EmissionMergeIntentFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                            config, emb_log)
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert']:
            emb_dim = 768
        else:
            emb_dim = opt.emb_dim

        # metric layer
        self.intent_metric = torch.nn.Linear(emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(emb_dim, opt.metric_dim)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        metric_sent_test_reps, metric_sent_support_reps = sent_test_reps, sent_support_reps  # intent
        metric_seq_test_reps, metric_seq_support_reps = seq_test_reps, seq_support_reps  # slot

        atv_lst = self.opt.metric_activation.split('-')
        intent_activation = atv_lst[0]
        slot_activation = atv_lst[0] if len(atv_lst) == 1 else atv_lst[1]

        if self.opt.split_metric in ['intent', 'both']:
            # intent metric space
            metric_sent_test_reps = self.intent_metric(metric_sent_test_reps)
            metric_sent_support_reps = self.intent_metric(metric_sent_support_reps)
            # activation
            if intent_activation == 'relu':
                metric_sent_test_reps = F.relu(metric_sent_test_reps)
                metric_sent_support_reps = F.relu(metric_sent_support_reps)
            elif intent_activation == 'sigmoid':
                metric_sent_test_reps = torch.sigmoid(metric_sent_test_reps)
                metric_sent_support_reps = torch.sigmoid(metric_sent_support_reps)
            elif intent_activation == 'tanh':
                metric_sent_test_reps = torch.tanh(metric_sent_test_reps)
                metric_sent_support_reps = torch.tanh(metric_sent_support_reps)
            elif intent_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        if self.opt.split_metric in ['slot', 'both']:
            # slot metric space
            metric_seq_test_reps = self.slot_metric(metric_seq_test_reps)
            metric_seq_support_reps = self.slot_metric(metric_seq_support_reps)
            # activation
            if slot_activation == 'relu':
                metric_seq_test_reps = F.relu(metric_seq_test_reps)
                metric_seq_support_reps = F.relu(metric_seq_support_reps)
            elif slot_activation == 'sigmoid':
                metric_seq_test_reps = torch.sigmoid(metric_seq_test_reps)
                metric_seq_support_reps = torch.sigmoid(metric_seq_support_reps)
            elif slot_activation == 'tanh':
                metric_seq_test_reps = torch.tanh(metric_seq_test_reps)
                metric_seq_support_reps = torch.tanh(metric_seq_support_reps)
            elif slot_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        ''' getting emission '''
        intent_emission = None
        slot_emission = None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(metric_sent_test_reps, intent_test_output_mask,
                                                               metric_sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(metric_seq_test_reps, slot_test_output_mask,
                                                           metric_seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)
        ''' merge intent emission to slot'''
        expand_intent_emission = self.expand_intent_emission_to_slot(slot_support_target, intent_support_target,
                                                                     self.slot_id2label, self.intent_id2label,
                                                                     intent_emission)
        # update emission with the one of the other task
        # update method is to concat them
        self.slot_decoder.cat_emission(expand_intent_emission)

        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                intent_loss = self.intent_decoder(metric_sent_test_reps, intent_test_output_mask, metric_sent_support_reps,
                                                  intent_support_output_mask, intent_test_target,
                                                  intent_support_target, self.label_mask)
                if self.opt.do_debug:
                    print('intent loss: {}'.format(intent_loss))
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss

            if self.learning_task in ['slot_filling', 'slu']:
                slot_loss = self.slot_decoder(metric_seq_test_reps, slot_test_output_mask, metric_seq_support_reps,
                                              slot_support_output_mask,
                                              slot_test_target, slot_support_target, self.label_mask)
                if self.opt.do_debug:
                    print('slot loss: {}'.format(slot_loss))
                loss += slot_loss
                self.loss_dict['slot'] = slot_loss

            return loss
        else:
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(metric_sent_test_reps, intent_test_output_mask,
                                                          metric_sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(metric_seq_test_reps, slot_test_output_mask,
                                                      metric_seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}


class EmissionMergeSlotFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None, ):
        super(EmissionMergeSlotFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                          config, emb_log)
        self.intent_id2label = self.opt.id2label['intent']
        self.slot_id2label = self.opt.id2label['slot']

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert']:
            emb_dim = 768
        else:
            emb_dim = opt.emb_dim

        # metric layer
        self.intent_metric = torch.nn.Linear(emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(emb_dim, opt.metric_dim)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        metric_sent_test_reps, metric_sent_support_reps = sent_test_reps, sent_support_reps  # intent
        metric_seq_test_reps, metric_seq_support_reps = seq_test_reps, seq_support_reps  # slot

        atv_lst = self.opt.metric_activation.split('-')
        intent_activation = atv_lst[0]
        slot_activation = atv_lst[0] if len(atv_lst) == 1 else atv_lst[1]

        if self.opt.split_metric in ['intent', 'both']:
            # intent metric space
            metric_sent_test_reps = self.intent_metric(metric_sent_test_reps)
            metric_sent_support_reps = self.intent_metric(metric_sent_support_reps)
            # activation
            if intent_activation == 'relu':
                metric_sent_test_reps = F.relu(metric_sent_test_reps)
                metric_sent_support_reps = F.relu(metric_sent_support_reps)
            elif intent_activation == 'sigmoid':
                metric_sent_test_reps = torch.sigmoid(metric_sent_test_reps)
                metric_sent_support_reps = torch.sigmoid(metric_sent_support_reps)
            elif intent_activation == 'tanh':
                metric_sent_test_reps = torch.tanh(metric_sent_test_reps)
                metric_sent_support_reps = torch.tanh(metric_sent_support_reps)
            elif intent_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        if self.opt.split_metric in ['slot', 'both']:
            # slot metric space
            metric_seq_test_reps = self.slot_metric(metric_seq_test_reps)
            metric_seq_support_reps = self.slot_metric(metric_seq_support_reps)
            # activation
            if slot_activation == 'relu':
                metric_seq_test_reps = F.relu(metric_seq_test_reps)
                metric_seq_support_reps = F.relu(metric_seq_support_reps)
            elif slot_activation == 'sigmoid':
                metric_seq_test_reps = torch.sigmoid(metric_seq_test_reps)
                metric_seq_support_reps = torch.sigmoid(metric_seq_support_reps)
            elif slot_activation == 'tanh':
                metric_seq_test_reps = torch.tanh(metric_seq_test_reps)
                metric_seq_support_reps = torch.tanh(metric_seq_support_reps)
            elif slot_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        ''' getting emission '''
        intent_emission = None
        slot_emission = None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(metric_sent_test_reps, intent_test_output_mask,
                                                               metric_sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(metric_seq_test_reps, slot_test_output_mask,
                                                           metric_seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)

        ''' merge slot emission to intent '''
        # expand emission as same as the one of the other task
        expand_slot_emission = self.expand_slot_emission_to_intent(slot_support_target, intent_support_target,
                                                                   self.slot_id2label, self.intent_id2label,
                                                                   slot_emission)
        # update emission with the one of the other task
        # update method is to concat them
        self.intent_decoder.cat_emission(expand_slot_emission)

        if self.training:
            loss = 0.
            if self.learning_task in ['slot_filling', 'slu']:
                intent_loss = self.slot_decoder(metric_seq_test_reps, slot_test_output_mask, metric_seq_support_reps,
                                                slot_support_output_mask,
                                                slot_test_target, slot_support_target, self.label_mask)
                if self.opt.do_debug:
                    print('intent loss: {}'.format(intent_loss))
                loss += intent_loss
                self.loss_dict['intent'] = intent_loss

            if self.learning_task in ['intent', 'slu']:
                slot_loss = self.intent_decoder(metric_sent_test_reps, intent_test_output_mask, metric_sent_support_reps,
                                                intent_support_output_mask, intent_test_target,
                                                intent_support_target, self.label_mask)
                if self.opt.do_debug:
                    print('slot loss: {}'.format(slot_loss))
                loss += slot_loss
                self.loss_dict['slot'] = slot_loss

            return loss
        else:
            intent_preds, slot_preds = None, None
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(metric_seq_test_reps, slot_test_output_mask,
                                                      metric_seq_support_reps,
                                                      slot_support_output_mask, slot_test_target,
                                                      slot_support_target, self.label_mask)
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(metric_sent_test_reps, intent_test_output_mask,
                                                          metric_sent_support_reps,
                                                          intent_support_output_mask, intent_test_target,
                                                          intent_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}


class SplitMetricFewShotSLU(FewShotSLU):

    def __init__(self,
                 opt,
                 context_embedder: ContextEmbedderBase,
                 intent_decoder: FewShotTextClassifier,
                 slot_decoder: FewShotSeqLabeler,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(SplitMetricFewShotSLU, self).__init__(opt, context_embedder, intent_decoder, slot_decoder,
                                                    config, emb_log)

        if opt.context_emb == 'electra':
            emb_dim = 256
        elif opt.context_emb in ['bert', 'sep_bert']:
            emb_dim = 768
        else:
            emb_dim = opt.emb_dim

        # metric layer
        self.intent_metric = torch.nn.Linear(emb_dim, opt.metric_dim)
        self.slot_metric = torch.nn.Linear(emb_dim, opt.metric_dim)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
    ):
        """
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: (batch_size, test_len)
        :param intent_test_output_mask: (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: (batch_size, support_size, support_len)
        :param intent_support_output_mask: (batch_size, support_size, support_len)
        :param slot_test_target: index targets (batch_size, test_len)
        :param slot_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: index targets (batch_size, test_len)
        :param intent_support_target: one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :return:
        """
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        ''' metric space controller '''
        metric_sent_test_reps, metric_sent_support_reps = sent_test_reps, sent_support_reps  # intent
        metric_seq_test_reps, metric_seq_support_reps = seq_test_reps, seq_support_reps  # slot

        atv_lst = self.opt.metric_activation.split('-')
        intent_activation = atv_lst[0]
        slot_activation = atv_lst[0] if len(atv_lst) == 1 else atv_lst[1]

        if self.opt.split_metric in ['intent', 'both']:
            # intent metric space
            metric_sent_test_reps = self.intent_metric(metric_sent_test_reps)
            metric_sent_support_reps = self.intent_metric(metric_sent_support_reps)
            # activation
            if intent_activation == 'relu':
                metric_sent_test_reps = F.relu(metric_sent_test_reps)
                metric_sent_support_reps = F.relu(metric_sent_support_reps)
            elif intent_activation == 'sigmoid':
                metric_sent_test_reps = torch.sigmoid(metric_sent_test_reps)
                metric_sent_support_reps = torch.sigmoid(metric_sent_support_reps)
            elif intent_activation == 'tanh':
                metric_sent_test_reps = torch.tanh(metric_sent_test_reps)
                metric_sent_support_reps = torch.tanh(metric_sent_support_reps)
            elif intent_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        if self.opt.split_metric in ['slot', 'both']:
            # slot metric space
            metric_seq_test_reps = self.slot_metric(metric_seq_test_reps)
            metric_seq_support_reps = self.slot_metric(metric_seq_support_reps)
            # activation
            if slot_activation == 'relu':
                metric_seq_test_reps = F.relu(metric_seq_test_reps)
                metric_seq_support_reps = F.relu(metric_seq_support_reps)
            elif slot_activation == 'sigmoid':
                metric_seq_test_reps = torch.sigmoid(metric_seq_test_reps)
                metric_seq_support_reps = torch.sigmoid(metric_seq_support_reps)
            elif slot_activation == 'tanh':
                metric_seq_test_reps = torch.tanh(metric_seq_test_reps)
                metric_seq_support_reps = torch.tanh(metric_seq_support_reps)
            elif slot_activation == 'none':
                pass
            else:
                raise TypeError('the metric_activation {} is not defined'.format(self.opt.metric_activation))

        ''' getting emission '''
        intent_emission = None
        slot_emission = None
        if self.opt.task in ['intent', 'slu']:
            intent_emission = self.intent_decoder.cal_emission(metric_sent_test_reps, intent_test_output_mask,
                                                               metric_sent_support_reps, intent_support_output_mask,
                                                               intent_test_target, intent_support_target)

        if self.opt.task in ['slot_filling', 'slu']:
            slot_emission = self.slot_decoder.cal_emission(metric_seq_test_reps, slot_test_output_mask,
                                                           metric_seq_support_reps, slot_support_output_mask,
                                                           slot_test_target, slot_support_target)

        if self.training:
            loss = 0.
            if self.learning_task in ['intent', 'slu']:
                loss += self.intent_decoder(metric_sent_test_reps, intent_test_output_mask, metric_sent_support_reps,
                                            intent_support_output_mask, intent_test_target,
                                            intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                loss += self.slot_decoder(metric_seq_test_reps, slot_test_output_mask, metric_seq_support_reps,
                                          slot_support_output_mask, slot_test_target, slot_support_target,
                                          self.label_mask)
            return loss
        else:
            intent_preds, slot_preds = None, None
            if self.learning_task in ['intent', 'slu']:
                intent_preds = self.intent_decoder.decode(metric_sent_test_reps, intent_test_output_mask,
                                                          metric_sent_support_reps,intent_support_output_mask,
                                                          intent_test_target, intent_support_target, self.label_mask)
            if self.learning_task in ['slot_filling', 'slu']:
                slot_preds = self.slot_decoder.decode(metric_seq_test_reps, slot_test_output_mask,
                                                      metric_seq_support_reps, slot_support_output_mask,
                                                      slot_test_target, slot_support_target, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}


class SchemaFewShotSLU(FewShotSLU):
    def __init__(
            self,
            opt,
            context_embedder: ContextEmbedderBase,
            model_map: Dict[str, torch.nn.Module],
            config: dict = None,  # store necessary setting or none-torch params
            emb_log: str = None
    ):
        super(SchemaFewShotSLU, self).__init__(opt, context_embedder, model_map, config, emb_log)

    def forward(
            self,
            test_token_ids: torch.Tensor,
            test_segment_ids: torch.Tensor,
            test_nwp_index: torch.Tensor,
            test_input_mask: torch.Tensor,
            slot_test_output_mask: torch.Tensor,
            intent_test_output_mask: torch.Tensor,
            support_token_ids: torch.Tensor,
            support_segment_ids: torch.Tensor,
            support_nwp_index: torch.Tensor,
            support_input_mask: torch.Tensor,
            slot_support_output_mask: torch.Tensor,
            intent_support_output_mask: torch.Tensor,
            slot_test_target: torch.Tensor,
            slot_support_target: torch.Tensor,
            intent_test_target: torch.Tensor,
            intent_support_target: torch.Tensor,
            support_num: torch.Tensor,
            slot_label_input: Tuple[torch.Tensor] = None,
            intent_label_input: Tuple[torch.Tensor] = None,
    ):
        """
        few-shot sequence labeler using schema information
        :param test_token_ids: (batch_size, test_len)
        :param test_segment_ids: (batch_size, test_len)
        :param test_nwp_index: (batch_size, test_len)
        :param test_input_mask: (batch_size, test_len)
        :param slot_test_output_mask: A dict of (batch_size, test_len)
        :param intent_test_output_mask: A dict of (batch_size, test_len)
        :param support_token_ids: (batch_size, support_size, support_len)
        :param support_segment_ids: (batch_size, support_size, support_len)
        :param support_nwp_index: (batch_size, support_size, support_len)
        :param support_input_mask: (batch_size, support_size, support_len)
        :param slot_support_output_mask: A dict of (batch_size, support_size, support_len)
        :param intent_support_output_mask: A dict of (batch_size, support_size, support_len)
        :param slot_test_target: A dict of index targets (batch_size, test_len)
        :param slot_support_target: A dict of one-hot targets (batch_size, support_size, support_len, num_tags)
        :param intent_test_target: A dict of index targets (batch_size, test_len)
        :param intent_support_target: A dict of one-hot targets (batch_size, support_size, support_len, num_tags)
        :param support_num: (batch_size, 1)
        :param slot_label_input: include
            label_token_ids: A tensor which is same to label token ids
                if label_reps=cat:
                    (batch_size, label_num * label_des_len)
                elif:
                    (batch_size, label_num, label_des_len)
            label_segment_ids: A tensor which is same to test token ids
            label_nwp_index: A tensor which is same to test token ids
            label_input_mask: A tensor which is same to label token ids
            label_output_mask: A tensor which is same to label token ids
        :param intent_label_input: include
        :return:
        """
        # reps for tokens: (batch_size, support_size, nwp_sent_len, emb_len)
        seq_test_reps, seq_support_reps, sent_test_reps, sent_support_reps = self.get_context_reps(
            test_token_ids, test_segment_ids, test_nwp_index, test_input_mask, support_token_ids, support_segment_ids,
            support_nwp_index, support_input_mask, self.opt.use_cls
        )

        slot_label_token_ids, slot_label_segment_ids, slot_label_nwp_index, slot_label_input_mask, _ = slot_label_input
        slot_label_reps = self.get_label_reps(slot_label_token_ids, slot_label_segment_ids, slot_label_nwp_index,
                                              slot_label_input_mask)
        intent_label_token_ids, intent_label_segment_ids, intent_label_nwp_index, intent_label_input_mask, _ = \
            intent_label_input
        intent_label_reps = self.get_label_reps(intent_label_token_ids, intent_label_segment_ids,
                                                intent_label_nwp_index, intent_label_input_mask)

        if self.training:
            loss = 0.
            loss += self.intent_decoder(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                        intent_support_output_mask, intent_test_target,
                                        intent_support_target, slot_label_reps, self.label_mask)
            loss += self.slot_decoder(seq_test_reps, slot_test_output_mask, seq_support_reps, slot_support_output_mask,
                                      slot_test_target, slot_support_target, intent_label_reps, self.label_mask)
            return loss
        else:
            intent_preds = self.intent_decoder.decode(sent_test_reps, intent_test_output_mask, sent_support_reps,
                                                      intent_support_output_mask, intent_test_target,
                                                      intent_support_target, slot_label_reps, self.label_mask)
            slot_preds = self.slot_decoder.decode(seq_test_reps, slot_test_output_mask, seq_support_reps,
                                                  slot_support_output_mask, slot_test_target,
                                                  slot_support_target, intent_label_reps, self.label_mask)
            return {'slot': slot_preds, 'intent': intent_preds}

    def get_label_reps(
            self,
            label_token_ids: torch.Tensor,
            label_segment_ids: torch.Tensor,
            label_nwp_index: torch.Tensor,
            label_input_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param label_token_ids:
        :param label_segment_ids:
        :param label_nwp_index:
        :param label_input_mask:
        :return:  shape (batch_size, label_num, label_des_len)
        """
        return self.context_embedder(
            label_token_ids, label_segment_ids, label_nwp_index, label_input_mask,  reps_type='label')

