# coding:utf-8
import collections
from typing import List, Tuple, Dict
import torch
import pickle
import json
import os
from transformers import BertTokenizer
from utils.data_loader import FewShotExample, DataItem
from utils.iter_helper import pad_tensor
from utils.config import SP_TOKEN_NO_MATCH, SP_LABEL_O, SP_TOKEN_O


FeatureItem = collections.namedtuple(   # text or raw features
    "FeatureItem",
    [
        "tokens",  # tokens corresponding to input token ids, eg: word_piece tokens with [CLS], [SEP]
        "slots",  # labels for all input position, eg; label for word_piece tokens
        "intent",  # labels for all input sentence
        "data_item",
        "token_ids",
        "segment_ids",
        "nwp_index",
        "input_mask",
        "slot_output_mask",
        "intent_output_mask",
    ]
)

ModelInput = collections.namedtuple(  # digit features for computation
    "ModelInput",   # all element shape: test: (1, test_len) support: (support_size, support_len)
    [
        "token_ids",  # token index list
        "segment_ids",  # bert [SEP] ids
        "nwp_index",  # non-word-piece word index to extract non-word-piece tokens' reps (only useful for bert).
        "input_mask",  # [1] * len(sent), 1 for valid (tokens, cls, sep, word piece), 0 is padding in batch construction
        "slot_output_mask",  # [1] * len(sent), 1 for valid output, 0 for padding, eg: 1 for original tokens in sl task
        "intent_output_mask",  # [1] * len(sent), 1 for valid output, 0 for padding, eg: 1 for original tokens in sl task
    ]
)


class FewShotFeature(object):
    """ pre-processed data for prediction """

    def __init__(
            self,
            gid: int,  # global id
            test_gid: int,
            batch_gid: int,
            test_input: ModelInput,
            test_feature_item: FeatureItem,
            support_input: ModelInput,
            support_feature_items: List[FeatureItem],
            slot_test_target: List[torch.Tensor] = None,
            slot_support_target: List[torch.Tensor] = None,
            intent_test_target: List[torch.Tensor] = None,
            intent_support_target: List[torch.Tensor] = None,
            slot_label_input: List[torch.Tensor] = None,
            slot_label_item: List[FeatureItem] = None,
            intent_label_input: List[torch.Tensor] = None,
            intent_label_item: List[FeatureItem] = None,
    ):
        """

        :param gid: global id
        :param test_gid: test data global id
        :param batch_gid: batch global id
        :param test_input: input
        :param test_feature_item:
        :param support_input:
        :param support_feature_items:
        :param slot_test_target:
        :param intent_test_target:
        :param slot_support_target:
        :param intent_support_target:
        :param slot_label_input:
        :param slot_label_item:
        :param intent_label_input:
        :param intent_label_item:
        """
        ''' identify info '''
        self.gid = gid
        self.test_gid = test_gid
        self.batch_gid = batch_gid
        ''' padded tensor for model '''
        self.test_input = test_input  # shape: (1, test_len)
        self.support_input = support_input  # shape: (support_size, support_len)
        # output:
        self.slot_test_target = slot_test_target
        self.slot_support_target = slot_support_target
        self.intent_test_target = intent_test_target
        self.intent_support_target = intent_support_target
        ''' raw feature '''
        self.test_feature_item = test_feature_item
        self.support_feature_items = support_feature_items
        self.slot_label_input = slot_label_input
        self.slot_label_item = slot_label_item
        self.intent_label_input = intent_label_input
        self.intent_label_item = intent_label_item

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__dict__)


class FewShotSupportFeature(object):
    """ pre-processed data for prediction """

    def __init__(
            self,
            gid: int,  # global id
            batch_gid: int,
            support_input: ModelInput,
            support_feature_items: FeatureItem,
            slot_support_target: torch.Tensor = None,
            intent_support_target: torch.Tensor = None,
            slot_label_input: torch.Tensor = None,
            slot_label_item: FeatureItem = None,
            intent_label_input: torch.Tensor = None,
            intent_label_item: FeatureItem = None,
    ):
        """

        :param gid: global id
        :param batch_gid: batch global id
        :param support_input:
        :param support_feature_items:
        :param slot_support_target:
        :param intent_support_target:
        :param slot_label_input:
        :param slot_label_item:
        :param intent_label_input:
        :param intent_label_item:
        """
        ''' identify info '''
        self.gid = gid
        self.batch_gid = batch_gid
        ''' padded tensor for model '''
        self.support_input = support_input  # shape: (support_len)
        # output:
        self.slot_support_target = slot_support_target
        self.intent_support_target = intent_support_target
        ''' raw feature '''
        self.support_feature_items = support_feature_items
        self.slot_label_input = slot_label_input
        self.slot_label_item = slot_label_item
        self.intent_label_input = intent_label_input
        self.intent_label_item = intent_label_item

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.__dict__)


class InputBuilderBase:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, example, max_support_size, slot_label2id, intent_label2id
                 ) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        raise NotImplementedError

    def data_item2feature_item(self, data_item: DataItem, seg_id: int) -> FeatureItem:
        raise NotImplementedError

    def get_test_model_input(self, feature_item: FeatureItem) -> ModelInput:
        ret = ModelInput(
            token_ids=torch.LongTensor(feature_item.token_ids),
            segment_ids=torch.LongTensor(feature_item.segment_ids),
            nwp_index=torch.LongTensor(feature_item.nwp_index),
            input_mask=torch.LongTensor(feature_item.input_mask),
            slot_output_mask=torch.LongTensor(feature_item.slot_output_mask),
            intent_output_mask=torch.LongTensor(feature_item.intent_output_mask),
        )
        return ret

    def get_support_model_input(self, feature_items: List[FeatureItem], max_support_size: int) -> ModelInput:
        pad_id = self.tokenizer.vocab['[PAD]']
        token_ids = self.pad_support_set([f.token_ids for f in feature_items], pad_id, max_support_size)
        segment_ids = self.pad_support_set([f.segment_ids for f in feature_items], 0, max_support_size)
        nwp_index = self.pad_support_set([f.nwp_index for f in feature_items], [0], max_support_size)
        input_mask = self.pad_support_set([f.input_mask for f in feature_items], 0, max_support_size)
        slot_output_mask = self.pad_support_set([f.slot_output_mask for f in feature_items], 0, max_support_size)
        intent_output_mask = self.pad_support_set([f.intent_output_mask for f in feature_items], 0, max_support_size)

        ret = ModelInput(
            token_ids=torch.LongTensor(token_ids),
            segment_ids=torch.LongTensor(segment_ids),
            nwp_index=torch.LongTensor(nwp_index),
            input_mask=torch.LongTensor(input_mask),
            slot_output_mask=torch.LongTensor(slot_output_mask),
            intent_output_mask=torch.LongTensor(intent_output_mask),
        )
        return ret

    def get_normal_model_input(self, feature_items: List[FeatureItem], max_support_size: int) -> List[ModelInput]:
        ret = []
        for f_item in feature_items:
            ret.append(ModelInput(
                token_ids=torch.LongTensor(f_item.token_ids),
                segment_ids=torch.LongTensor(f_item.segment_ids),
                nwp_index=torch.LongTensor(f_item.nwp_index),
                input_mask=torch.LongTensor(f_item.input_mask),
                slot_output_mask=torch.LongTensor(f_item.slot_output_mask),
                intent_output_mask=torch.LongTensor(f_item.intent_output_mask),
            ))
        return ret

    def pad_support_set(self, item_lst: List[List[int]], pad_value: int, max_support_size: int) -> List[List[int]]:
        """
        pre-pad support set to insure: 1. each spt set has same sent num 2. each sent has same length
        (do padding here because: 1. all support sent are considered as one tensor input  2. support set size is small)
        :param item_lst:
        :param pad_value:
        :param max_support_size:
        :return:
        """
        ''' pad sentences '''
        max_sent_len = max([len(x) for x in item_lst])  # max length among one
        ret = []
        for sent in item_lst:
            temp = sent[:]
            while len(temp) < max_sent_len:
                temp.append(pad_value)
            ret.append(temp)
        ''' pad support set size '''
        pad_item = [pad_value for _ in range(max_sent_len)]
        while len(ret) < max_support_size:
            ret.append(pad_item)
        return ret

    def digitizing_input(self, tokens: List[str], seg_id: int) -> (List[int], List[int]):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [seg_id for _ in range(len(tokens))]
        return token_ids, segment_ids

    def tokenizing(self, item: DataItem):
        """ Possible tokenizing for item """
        pass


class BertInputBuilder(InputBuilderBase):
    def __init__(self, tokenizer, opt):
        super(BertInputBuilder, self).__init__(tokenizer)
        self.opt = opt
        self.test_seg_id = 0
        self.support_seg_id = 0 if opt.context_emb == 'sep_bert' else 1  # 1 to cat support and query to get reps
        self.seq_ins = {}

    def __call__(self, example, max_support_size, slot_label2id, intent_label2id
                 ) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item, test_input = self.prepare_test(example)
        support_feature_items, support_input = self.prepare_support(example, max_support_size)
        return test_feature_item, test_input, support_feature_items, support_input

    def prepare_test(self, example):
        test_feature_item = self.data_item2feature_item(data_item=example.test_data_item, seg_id=0)
        test_input = self.get_test_model_input(test_feature_item)
        return test_feature_item, test_input

    def prepare_support(self, example, max_support_size):
        support_feature_items = [self.data_item2feature_item(data_item=s_item, seg_id=self.support_seg_id)
                                 for s_item in example.support_data_items]
        support_input = self.get_support_model_input(support_feature_items, max_support_size)
        return support_feature_items, support_input

    def data_item2feature_item(self, data_item: DataItem, seg_id: int) -> FeatureItem:
        """ get feature_item for bert, steps: 1. do digitalizing 2. make mask """
        wp_mark, wp_text = self.tokenizing(data_item)
        slots = self.get_wp_label(data_item.seq_out, wp_text, wp_mark) if self.opt.label_wp else data_item.seq_out
        slot_output_mask = [1] * len(slots)
        intent = data_item.label
        if isinstance(intent, str):
            intent = [intent]
        intent_output_mask = [1] * len(intent)

        tokens = ['[CLS]'] + wp_text + ['[SEP]'] if seg_id == 0 else wp_text + ['[SEP]']
        token_ids, segment_ids = self.digitizing_input(tokens=tokens, seg_id=seg_id)
        nwp_index = self.get_nwp_index(wp_mark)
        input_mask = [1] * len(token_ids)
        ret = FeatureItem(
            tokens=tokens,
            slots=slots,
            intent=intent,
            data_item=data_item,
            token_ids=token_ids,
            segment_ids=segment_ids,
            nwp_index=nwp_index,
            input_mask=input_mask,
            slot_output_mask=slot_output_mask,
            intent_output_mask=intent_output_mask,
        )
        return ret

    def get_nwp_index(self, word_piece_mark: list) -> torch.Tensor:
        """ get index of non-word-piece tokens, which is used to extract non-wp bert embedding in batch manner """
        return torch.nonzero(torch.LongTensor(word_piece_mark) - 1).tolist()  # wp mark word-piece with 1, so - 1

    def tokenizing(self, item: DataItem):
        """ Do tokenizing and get word piece data and get label on pieced words. """
        wp_text = self.tokenizer.wordpiece_tokenizer.tokenize(' '.join(item.seq_in))
        wp_mark = [int((len(w) > 2) and w[0] == '#' and w[1] == '#') for w in wp_text]  # mark wp as 1
        return wp_mark, wp_text

    def get_wp_label(self, label_lst, wp_text, wp_mark, label_pieced_words=False):
        """ get label on pieced words. """
        wp_label, label_idx = [], 0
        for ind, mark in enumerate(wp_mark):
            if mark == 0:  # label non-pieced token with original label
                wp_label.append(label_lst[label_idx])
                label_idx += 1  # pointer on non-wp labels
            elif mark == 1:  # label word-piece with whole word's label or with  [PAD] label
                pieced_label = wp_label[-1].replace('B-', 'I-') if label_pieced_words else '[PAD]'
                wp_label.append(pieced_label)
            if not wp_label[-1]:
                raise RuntimeError('Empty label')
        if not (len(wp_label) == len(wp_text) == len(wp_mark)):
            raise RuntimeError('ERROR: Failed to generate wp labels:{}{}{}{}{}{}{}{}{}{}{}'.format(
                len(wp_label), len(wp_text), len(wp_mark),
                '\nwp_lb', wp_label, '\nwp_text', wp_text, '\nwp_mk', wp_mark, '\nlabel', label_lst))


class SupportInputBuilder(BertInputBuilder):
    def __init__(self, tokenizer, opt):
        super(SupportInputBuilder, self).__init__(tokenizer, opt)

    def __call__(self, example, max_support_size, slot_label2id, intent_label2id
                 ) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        normal_feature_items, normal_input = self.prepare_support(example, max_support_size)
        if self.opt.ft_add_query:
            test_feature_items, test_input = self.prepare_test(example, max_support_size)
            normal_feature_items.extend(test_feature_items)
            normal_input.extend(test_input)

        return normal_feature_items, normal_input

    def prepare_test(self, example):
        test_feature_item = [self.data_item2feature_item(data_item=example.test_data_item, seg_id=0)]
        test_input = self.get_normal_model_input(test_feature_item)
        return test_feature_item, test_input

    def prepare_support(self, example, max_support_size):
        support_feature_items = [self.data_item2feature_item(data_item=s_item, seg_id=0)  # support data is also with [CLS]
                                 for s_item in example.support_data_items]
        support_input = self.get_normal_model_input(support_feature_items, max_support_size)
        return support_feature_items, support_input


class SchemaInputBuilder(BertInputBuilder):
    def __init__(self, tokenizer, opt):
        super(SchemaInputBuilder, self).__init__(tokenizer, opt)

    def __call__(self, example, max_support_size, slot_label2id, intent_label2id
                 ) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item, test_input = self.prepare_test(example)
        support_feature_items, support_input = self.prepare_support(example, max_support_size)
        if self.opt.label_reps in ['cat']:  # represent labels by concat all all labels
            label_input, label_items = self.prepare_label_feature(slot_label2id, intent_label2id)
        elif self.opt.label_reps in ['sep', 'sep_sum']:  # represent each label independently
            label_input, label_items = self.prepare_sep_label_feature(slot_label2id, intent_label2id)
        else:
            raise TypeError('the label_reps should be one of cat & set & sep_num')
        return test_feature_item, test_input, support_feature_items, support_input, label_items, label_input,

    def prepare_label_feature(self, slot_label2id: Dict[str, int], intent_label2id: Dict[str, int]):
        """ prepare digital input for label feature in concatenate style """
        slot_sorted_labels = sorted(slot_label2id.items(), key=lambda x: x[1])
        intent_sorted_labels = sorted(intent_label2id.items(), key=lambda x: x[1])

        seq_ins, seq_outs, labels = [], [], []
        for label_name, label_id in slot_sorted_labels:
            if label_name == '[PAD]':
                continue
            tmp_text = self.convert_label_name(label_name)
            seq_ins.extend(tmp_text)
            seq_outs.extend(['O'] * len(tmp_text))
            labels.extend(['O'])
        slot_label_item = self.data_item2feature_item(DataItem(seq_in=seq_ins, seq_out=seq_outs, label=labels), 0)
        slot_label_input = self.get_test_model_input(slot_label_item)

        seq_ins, seq_outs, labels = [], [], []
        for label_name, label_id in intent_sorted_labels:
            if label_name == '[PAD]':
                continue
            tmp_text = self.convert_label_name(label_name)
            seq_ins.extend(tmp_text)
            seq_outs.extend(['O'] * len(tmp_text))
            labels.extend(['O'])
        intent_label_item = self.data_item2feature_item(DataItem(seq_in=seq_ins, seq_out=seq_outs, label=labels), 0)
        intent_label_input = self.get_test_model_input(intent_label_item)

        return (slot_label_input, slot_label_item), (intent_label_input, intent_label_item)

    def prepare_sep_label_feature(self, slot_label2id, intent_label2id):
        """ prepare digital input for label feature separately """
        slot_label_item = []
        for label_name in slot_label2id:
            if label_name == '[PAD]':
                continue
            seq_in = self.convert_label_name(label_name)
            seq_out = ['None'] * len(seq_in)
            label = ['None']
            slot_label_item.append(self.data_item2feature_item(DataItem(seq_in, seq_out, label), 0))
        slot_label_input = self.get_support_model_input(slot_label_item, len(slot_label2id) - 1)

        intent_label_item = []
        for label_name in intent_label2id:
            if label_name == '[PAD]':
                continue
            seq_in = self.convert_label_name(label_name)
            seq_out = ['None'] * len(seq_in)
            label = ['None']
            slot_label_item.append(self.data_item2feature_item(DataItem(seq_in, seq_out, label), 0))
        intent_label_input = self.get_support_model_input(slot_label_item, len(slot_label2id) - 1)

        return (slot_label_input, slot_label_item), (intent_label_input, intent_label_item)

    def convert_label_name(self, name):
        text = []
        tmp_name = name
        if 'B-' in name:
            text.append('begin')
            tmp_name = name.replace('B-', '')
        elif 'I-' in name:
            text.append('inner')
            tmp_name = name.replace('I-', '')
        elif 'O' == name:
            text.append('ordinary')
            tmp_name = ''

        # special processing to label name
        name_translations = [('PER', 'person'), ('ORG', 'organization'), ('LOC', 'location'),
                             ('MISC', 'miscellaneous'), ('GPE', 'geographical political'),
                             ('NORP', 'nationalities or religious or political groups'),
                             # toursg data
                             ("ACK", "acknowledgment, as well as common expressions used for grounding"),
                             # ("CANCEL", "cancelation"),
                             # ("CLOSING", "closing remarks"),
                             # ("COMMIT", "commitment"),
                             # ("CONFIRM", "confirmation"),
                             # ("ENOUGH", "no more information is needed"),
                             # ("EXPLAIN", "an explanation/justification of a previous stated idea"),
                             # ("HOW_MUCH", "money or time amounts"),
                             # ("HOW_TO", "used to request/give specific instructions"),
                             ("INFO", "information request"),
                             # ("NEGATIVE", "negative responses"),
                             # ("OPENING", "opening remarks"),
                             # ("POSITIVE", "positive responses"),
                             # ("PREFERENCE", "preferences"),
                             # ("RECOMMEND", "recommendations"),
                             # ("THANK", "thank you remarks"),
                             # ("WHAT", "concept related utterances"),
                             # ("WHEN", "time related utterances"),
                             # ("WHERE", "location related utterances"),
                             # ("WHICH", "entity related utterances"),
                             # ("WHO", "person related utterances and questions"),
                             ]
        if tmp_name:
            for shot, long in name_translations:
                if tmp_name == shot:
                    text.append(long)
                    tmp_name = ''
                    break
        if tmp_name:
            text.extend(tmp_name.lower().split('_'))
        return text


class NormalInputBuilder(InputBuilderBase):
    def __init__(self, tokenizer, opt):
        super(NormalInputBuilder, self).__init__(tokenizer)
        self.opt = opt

    def __call__(self, example, max_support_size, label2id) -> (FeatureItem, ModelInput, List[FeatureItem], ModelInput):
        test_feature_item = self.data_item2feature_item(data_item=example.test_data_item, seg_id=0)
        test_input = self.get_test_model_input(test_feature_item)
        support_feature_items = [self.data_item2feature_item(data_item=s_item, seg_id=1) for s_item in
                                 example.support_data_items]
        support_input = self.get_support_model_input(support_feature_items, max_support_size)
        return test_feature_item, test_input, support_feature_items, support_input

    def data_item2feature_item(self, data_item: DataItem, seg_id: int) -> FeatureItem:
        """ get feature_item for bert, steps: 1. do padding 2. do digitalizing 3. make mask """
        tokens = data_item.seq_in

        slots = data_item.seq_out
        slot_output_mask = [1] * len(slots)
        intent = data_item.label
        if isinstance(intent, str):
            intent = [intent]
        intent_output_mask = [1] * len(intent)

        token_ids, segment_ids = self.digitizing_input(tokens=tokens, seg_id=seg_id)
        nwp_index = [[i] for i in range(len(token_ids))]
        input_mask = [1] * len(token_ids)
        ret = FeatureItem(
            tokens=tokens,
            slots=slots,
            intent=intent,
            data_item=data_item,
            token_ids=token_ids,
            segment_ids=segment_ids,
            nwp_index=nwp_index,
            input_mask=input_mask,
            slot_output_mask=slot_output_mask,
            intent_output_mask=intent_output_mask,
        )
        return ret


class OutputBuilderBase:
    """  Digitalizing the output targets"""
    def __init__(self):
        pass

    def __call__(self, test_feature_item: FeatureItem, support_feature_items: FeatureItem,
                 slot_label2id: dict, intent_label2id: dict, max_support_size: int):
        raise NotImplementedError

    def pad_support_set(self, item_lst: List[List[List[int]]], pad_value: List[int], max_support_size: int) -> List[List[int]]:
        """
        pre-pad support set to insure: 1. each set has same sent num 2. each sent has same length
        (do padding here because: 1. all support sent are considered as one tensor input  2. support set size is small)
        :param item_lst:
        :param pad_value:
        :param max_support_size:
        :return:
        """
        ''' pad sentences '''
        max_sent_len = max([len(x) for x in item_lst])
        ret = []
        for sent in item_lst:
            temp = sent[:]
            while len(temp) < max_sent_len:
                temp.append(pad_value)
            ret.append(temp)
        ''' pad support set size '''
        pad_item = [pad_value for _ in range(max_sent_len)]
        while len(ret) < max_support_size:
            ret.append(pad_item)
        return ret


class FewShotOutputBuilder(OutputBuilderBase):
    """  Digitalizing the output targets as label id for non word piece tokens  """
    def __init__(self):
        super(FewShotOutputBuilder, self).__init__()

    def __call__(self, test_feature_item: FeatureItem, support_feature_items: FeatureItem,
                 slot_label2id: Dict[str, int], intent_label2id: Dict[str, int], max_support_size: int):
        slot_test_target = self.item2label_ids(test_feature_item, slot_label2id, 'slot')
        slot_support_target = [self.item2label_onehot(f_item, slot_label2id, 'slot')
                               for f_item in support_feature_items]
        slot_support_target = self.pad_support_set(slot_support_target, self.label2onehot('[PAD]', slot_label2id),
                                                   max_support_size)

        intent_test_target = self.item2label_ids(test_feature_item, intent_label2id, 'intent')
        intent_support_target = [self.item2label_onehot(f_item, intent_label2id, 'intent')
                                 for f_item in support_feature_items]
        intent_support_target = self.pad_support_set(intent_support_target, self.label2onehot('[PAD]', intent_label2id),
                                                     max_support_size)

        return (torch.LongTensor(slot_test_target), torch.LongTensor(slot_support_target)), \
               (torch.LongTensor(intent_test_target), torch.LongTensor(intent_support_target))

    def item2label_ids(self, f_item: FeatureItem, label2id: dict, task: str):
        item = f_item.slots if task == 'slot' else f_item.intent
        # print('{} - item: {}'.format(task, item))
        return [label2id[lb] for lb in item]

    def item2label_onehot(self, f_item: FeatureItem, label2id: dict, task):
        item = f_item.slots if task == 'slot' else f_item.intent
        return [self.label2onehot(lb, label2id) for lb in item]

    def label2onehot(self, label: str, label2id: dict):
        onehot = [0 for _ in range(len(label2id))]
        onehot[label2id[label]] = 1
        return onehot


class SupportOutputBuilder(OutputBuilderBase):
    """  Digitalizing the output targets as label id for non word piece tokens  """
    def __init__(self):
        super(SupportOutputBuilder, self).__init__()

    def __call__(self, test_feature_item: None, support_feature_item: FeatureItem,
                 slot_label2id: Dict[str, int], intent_label2id: Dict[str, int], max_support_size: int):
        slot_support_target = self.item2label_ids(support_feature_item, slot_label2id, 'slot')

        intent_support_target = self.item2label_ids(support_feature_item, intent_label2id, 'intent')

        return torch.LongTensor(slot_support_target), torch.LongTensor(intent_support_target)

    def item2label_ids(self, f_item: FeatureItem, label2id: dict, task: str):
        item = f_item.slots if task == 'slot' else f_item.intent
        return [label2id[lb] for lb in item]


class FeatureConstructor:
    """
    Class for build feature and label2id dict
    Main function:
        construct_feature
        make_dict
    """
    def __init__(
            self,
            input_builder: InputBuilderBase,
            output_builder: OutputBuilderBase,
    ):
        self.input_builder = input_builder
        self.output_builder = output_builder

    def construct_feature(
            self,
            examples: List[FewShotExample],
            max_support_size: int,
            slot_label2id: Dict[str, int],
            slot_id2label: Dict[int, str],
            intent_label2id: Dict[str, int],
            intent_id2label: Dict[int, str],
    ) -> List[FewShotFeature]:
        all_features = []
        for example in examples:
            feature = self.example2feature(example, max_support_size,
                                           slot_label2id, slot_id2label, intent_label2id, intent_id2label)
            all_features.append(feature)
        return all_features

    def example2feature(
            self,
            example: FewShotExample,
            max_support_size: int,
            slot_label2id: Dict[str, int],
            slot_id2label: Dict[int, str],
            intent_label2id: Dict[str, int],
            intent_id2label: Dict[int, str],
    ) -> FewShotFeature:
        test_feature_item, test_input, support_feature_items, support_input = self.input_builder(
            example, max_support_size, slot_label2id, intent_label2id)
        (slot_test_target, slot_support_target), (intent_test_target, intent_support_target) = self.output_builder(
            test_feature_item, support_feature_items, slot_label2id, intent_label2id, max_support_size)
        ret = FewShotFeature(
            gid=example.gid,
            test_gid=example.test_id,
            batch_gid=example.batch_id,
            test_input=test_input,
            test_feature_item=test_feature_item,
            support_input=support_input,
            support_feature_items=support_feature_items,
            slot_test_target=slot_test_target,
            slot_support_target=slot_support_target,
            intent_test_target=intent_test_target,
            intent_support_target=intent_support_target,
        )
        return ret


class SupportFeatureConstructor:
    """
    Class for build feature and label2id dict
    Main function:
        construct_feature
        make_dict
    """
    def __init__(
            self,
            input_builder: SupportInputBuilder,
            output_builder: SupportOutputBuilder,
    ):
        self.input_builder = input_builder
        self.output_builder = output_builder
        self.gid = 0
        self.text_set = set()

    def construct_feature(
            self,
            examples: List[FewShotExample],
            max_support_size: int,
            slot_label2id: Dict[str, int],
            slot_id2label: Dict[int, str],
            intent_label2id: Dict[str, int],
            intent_id2label: Dict[int, str],
    ) -> List[FewShotSupportFeature]:
        self.gid = 0
        self.text_set = set()
        all_features = []
        for example in examples:
            feature = self.example2feature(example, max_support_size,
                                           slot_label2id, slot_id2label, intent_label2id, intent_id2label)
            all_features.extend(feature)
        return all_features

    def example2feature(
            self,
            example: FewShotExample,
            max_support_size: int,
            slot_label2id: Dict[str, int],
            slot_id2label: Dict[int, str],
            intent_label2id: Dict[str, int],
            intent_id2label: Dict[int, str],
    ) -> List[FewShotSupportFeature]:
        support_feature_items, support_input = self.input_builder(example, max_support_size,
                                                                  slot_label2id, intent_label2id)
        ret = []
        for support_feature_item, support_input_item in zip(support_feature_items, support_input):
            slot_support_target, intent_support_target = self.output_builder(None, support_feature_item, slot_label2id,
                                                                             intent_label2id, max_support_size)
            text = ''.join(support_feature_item.tokens)
            if text not in self.text_set:
                ret.append(FewShotSupportFeature(
                    gid=self.gid,
                    batch_gid=example.batch_id,
                    support_input=support_input_item,
                    support_feature_items=support_feature_item,
                    slot_support_target=slot_support_target,
                    intent_support_target=intent_support_target,
                ))
                self.gid += 1
        return ret


class SchemaFeatureConstructor(FeatureConstructor):
    def __init__(
            self,
            input_builder: InputBuilderBase,
            output_builder: OutputBuilderBase,
    ):
        super(SchemaFeatureConstructor, self).__init__(input_builder, output_builder)

    def example2feature(
            self,
            example: FewShotExample,
            max_support_size: int,
            slot_label2id: Dict[str, int],
            slot_id2label: Dict[int, str],
            intent_label2id: Dict[str, int],
            intent_id2label: Dict[int, str],
    ) -> FewShotFeature:
        test_feature_item, test_input, support_feature_items, support_input, \
        (slot_label_input, slot_label_item), (intent_label_input, intent_label_item) = \
            self.input_builder(example, max_support_size, slot_label2id, intent_label2id)
        slot_test_target, slot_support_target, intent_test_target, intent_support_target = self.output_builder(
            test_feature_item, support_feature_items, slot_label2id, intent_label2id, max_support_size)
        ret = FewShotFeature(
            gid=example.gid,
            test_gid=example.test_id,
            batch_gid=example.batch_id,
            test_input=test_input,
            test_feature_item=test_feature_item,
            support_input=support_input,
            support_feature_items=support_feature_items,
            slot_test_target=slot_test_target,
            slot_support_target=slot_support_target,
            intent_test_target=intent_test_target,
            intent_support_target=intent_support_target,
            slot_label_input=slot_label_input,
            slot_label_item=slot_label_item,
            intent_label_input=intent_label_input,
            intent_label_item=intent_label_item,
        )
        return ret


def flatten(l):
    """ convert list of list to list"""
    return [item for sublist in l for item in sublist]


def make_dict(opt, examples: List[FewShotExample]) -> (Dict[str, int], Dict[int, str]):
    """
    make label2id dict
    label2id must follow rules:
    For sequence labeling:
        1. id(PAD)=0 id(O)=1  2. id(B-X)=i  id(I-X)=i+1
    For (multi-label) text classification:
        1. id(PAD)=0
    """
    def purify(l):
        """ remove B- and I- """
        return set([item.replace('B-', '').replace('I-', '') for item in l])

    ''' collect all label from: all test set & all support set '''
    all_slots, all_intent = [], []
    slot_label2id, intent_label2id = {'[PAD]': 0}, {'[PAD]': 0}
    for example in examples:
        all_slots.append(example.test_data_item.seq_out)
        all_slots.extend([data_item.seq_out for data_item in example.support_data_items])

        test_intent = example.test_data_item.label if isinstance(example.test_data_item.label, list) \
            else [example.test_data_item.label]  # transfer the label to a list for fitting our code
        all_intent.append(test_intent)
        all_intent.extend([data_item.label if isinstance(data_item.label, list) else [data_item.label]
                           for data_item in example.support_data_items])
    ''' collect label word set '''
    # sort to make embedding id fixed
    slots_set = sorted(list(purify(set(flatten(all_slots)))))
    intent_set = sorted(list(purify(set(flatten(all_intent)))))
    ''' build dict '''
    slot_label2id['O'] = len(slot_label2id)
    for label in slots_set:
        if label == 'O':
            continue
        slot_label2id['B-' + label] = len(slot_label2id)
        slot_label2id['I-' + label] = len(slot_label2id)

    for label in intent_set:
        intent_label2id[label] = len(intent_label2id)

    ''' reverse the label2id '''
    slot_id2label = dict([(idx, label) for label, idx in slot_label2id.items()])
    intent_id2label = dict([(idx, label) for label, idx in intent_label2id.items()])

    return (slot_label2id, slot_id2label), (intent_label2id, intent_id2label)


def make_word_dict(all_files: List[str]) -> (Dict[str, int], Dict[int, str]):
    all_words = []
    word2id = {}
    for file in all_files:
        with open(file, 'r') as reader:
            raw_data = json.load(reader)
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            for batch_id, batch in enumerate(domain):
                all_words.extend(batch['support']['seq_ins'])
                all_words.extend(batch['query']['seq_ins'])
    word_set = sorted(list(set(flatten(all_words))))  # sort to make embedding id fixed
    for word in ['[PAD]', '[OOV]'] + word_set:
        word2id[word] = len(word2id)
    id2word = dict([(idx, word) for word, idx in word2id.items()])
    return word2id, id2word


def make_mask(token_ids: torch.Tensor, label_ids: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    input_mask = (token_ids != 0).long()
    output_mask = (label_ids != 0).long()  # mask
    return input_mask, output_mask


def save_feature(path, features, slot_label2id, slot_id2label, intent_label2id, intent_id2label):
    with open(path, 'wb') as writer:
        saved_features = {
            'features': features,
            'slot_label2id': slot_label2id,
            'slot_id2label': slot_id2label,
            'intent_label2id': intent_label2id,
            'intent_id2label': intent_id2label,
        }
        pickle.dump(saved_features, writer)


def load_feature(path):
    with open(path, 'rb') as reader:
        saved_feature = pickle.load(reader)
        return saved_feature['features'], (saved_feature['slot_label2id'], saved_feature['slot_id2label']), \
            (saved_feature['intent_label2id'], saved_feature['intent_id2label'])


def make_preprocessor(opt, finetune=False):
    """ make preprocessor """
    transformer_style_embs = ['bert', 'sep_bert', 'electra']

    ''' select input_builder '''
    if opt.context_emb not in transformer_style_embs:
        word2id, id2word = make_word_dict([opt.train_path, opt.dev_path, opt.test_path])
        opt.word2id = word2id

    if opt.context_emb in transformer_style_embs:
        tokenizer = BertTokenizer.from_pretrained(opt.bert_vocab)
        if opt.use_schema:
            input_builder = SchemaInputBuilder(tokenizer=tokenizer, opt=opt)
        else:
            if not finetune:
                input_builder = BertInputBuilder(tokenizer=tokenizer, opt=opt)
            else:
                input_builder = SupportInputBuilder(tokenizer=tokenizer, opt=opt)

    elif opt.context_emb == 'elmo':
        raise NotImplementedError
    elif opt.context_emb in ['glove', 'raw']:
        tokenizer = MyTokenizer(word2id=word2id, id2word=id2word)
        input_builder = NormalInputBuilder(tokenizer=tokenizer)
    else:
        raise TypeError('wrong word representation type')

    ''' select output builder '''
    if not finetune:
        output_builder = FewShotOutputBuilder()
    else:
        output_builder = SupportOutputBuilder()

    ''' build preprocessor '''
    if opt.use_schema:
        preprocessor = SchemaFeatureConstructor(input_builder=input_builder, output_builder=output_builder)
    else:
        if not finetune:
            preprocessor = FeatureConstructor(input_builder=input_builder, output_builder=output_builder)
        else:
            preprocessor = SupportFeatureConstructor(input_builder=input_builder, output_builder=output_builder)
    return preprocessor


def make_label_mask(opt, path, label2id):
    """ disable cross domain transition """
    label_mask = [[0] * len(label2id) for _ in range(len(label2id))]
    with open(path, 'r') as reader:
        raw_data = json.load(reader)
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            batch = domain[0]
            supports_labels = batch['support']['seq_outs']
            all_support_labels = set(collections._chain.from_iterable(supports_labels))
            for lb_from in all_support_labels:
                for lb_to in all_support_labels:
                    if opt.do_debug:  # when debuging, only part of labels are leveraged
                        if lb_from not in label2id or lb_to not in label2id:
                            continue
                    label_mask[label2id[lb_from]][label2id[lb_to]] = 1
    return torch.LongTensor(label_mask)


class MyTokenizer(object):
    def __init__(self, word2id, id2word):
        self.word2id = word2id
        self.id2word = id2word
        self.vocab = word2id

    def convert_tokens_to_ids(self, tokens):
        return [self.word2id[token] for token in tokens]
