# coding:utf-8
import json
import copy
import collections
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, Sampler


class RawDataLoaderBase:
    def __init__(self, *args, **kwargs):
        pass

    def load_data(self, path: str):
        pass


DataItem = collections.namedtuple("DataItem", ["seq_in", "seq_out", "label"])


class FewShotExample(object):
    """  Each few-shot example is a pair of (one query example, support set) """

    def __init__(
            self,
            gid: int,
            batch_id: int,
            test_id: int,
            domain_name: str,
            support_data_items: List[DataItem],
            test_data_item: DataItem
    ):
        self.gid = gid
        self.batch_id = batch_id
        self.test_id = test_id  # query relative index in one episode
        self.domain_name = domain_name

        self.support_data_items = support_data_items  # all support data items
        self.test_data_item = test_data_item  # one query data items

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'gid:{}\n\tdomain:{}\n\ttest_data:{}\n\ttest_label:{}\n\tsupport_data:{}'.format(
            self.gid,
            self.domain_name,
            self.test_data_item.seq_in,
            self.test_data_item.seq_out,
            self.support_data_items,
        )


class FewShotRawDataLoader(RawDataLoaderBase):
    def __init__(self, opt):
        super(FewShotRawDataLoader, self).__init__()
        self.opt = opt
        self.debugging = opt.do_debug
        self.idx_dict = {'O': 0, 'B': 1, 'I': 2}

    def load_data(self, path: str) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
            load few shot data set
            input:
                path: file path
            output
                examples: a list, all example loaded from path
                few_shot_batches: a list, of fewshot batch, each batch is a list of examples
                max_len: max sentence length
            """
        with open(path, 'r') as reader:
            raw_data = json.load(reader)
            examples, few_shot_batches, max_support_size = self.raw_data2examples(raw_data)
        if self.debugging:
            examples, few_shot_batches = examples[:8], few_shot_batches[:2]
        return examples, few_shot_batches, max_support_size

    def raw_data2examples(self, raw_data: Dict) -> (List[FewShotExample], List[List[FewShotExample]], int):
        """
        process raw_data into examples
        """
        examples = []
        all_support_size = []
        few_shot_batches = []
        trans_mat = torch.zeros(3, 5, dtype=torch.int32).tolist()
        start_trans_mat = torch.zeros(3, dtype=torch.int32).tolist()
        end_trans_mat = torch.zeros(3, dtype=torch.int32).tolist()
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            for batch_id, batch in enumerate(domain):
                one_batch_examples = []
                support_data_items, test_data_items = self.batch2data_items(batch)
                # update transition matrix
                self.get_trans_mat(trans_mat, start_trans_mat, end_trans_mat, support_data_items)

                all_support_size.append(len(support_data_items))
                ''' Pair each test sample with full support set '''
                for test_id, test_data_item in enumerate(test_data_items):
                    gid = len(examples)
                    example = FewShotExample(
                        gid=gid,
                        batch_id=batch_id,
                        test_id=test_id,
                        domain_name=domain_n,
                        test_data_item=test_data_item,
                        support_data_items=support_data_items,
                    )
                    examples.append(example)
                    one_batch_examples.append(example)
                few_shot_batches.append(one_batch_examples)
        max_support_size = max(all_support_size)
        return examples, few_shot_batches, max_support_size

    def batch2data_items(self, batch: dict) -> (List[DataItem], List[DataItem]):
        support_data_items = self.get_data_items(parts=batch['support'])
        test_data_items = self.get_data_items(parts=batch['query'])
        return support_data_items, test_data_items

    def get_data_items(self, parts: dict) -> List[DataItem]:
        data_item_lst = []
        for seq_in, seq_out, label in zip(parts['seq_ins'], parts['seq_outs'], parts['labels']):
            data_item = DataItem(seq_in=seq_in, seq_out=seq_out, label=label)
            data_item_lst.append(data_item)
        return data_item_lst

    def get_trans_mat(self,
                      trans_mat: List[List[int]],
                      start_trans_mat: List[int],
                      end_trans_mat: List[int],
                      support_data: List[str]) -> None:
        for support_data_item in support_data:
            labels = support_data_item.label
            s_idx = self.idx_dict[labels[0][0]]
            e_idx = self.idx_dict[labels[-1][0]]
            start_trans_mat[s_idx] += 1
            end_trans_mat[e_idx] += 1
            for i in range(len(labels) - 1):
                cur_label = labels[i]
                next_label = labels[i + 1]
                start_idx = self.idx_dict[cur_label[0]]
                if cur_label == next_label:
                    end_idx = self.idx_dict[next_label[0]]
                else:
                    if cur_label[0] == next_label[0]:
                        end_idx = self.idx_dict[next_label[0]] + 2
                    else:
                        if cur_label == 'O':
                            end_idx = self.idx_dict[next_label[0]]
                        elif next_label == 'O':
                            end_idx = 0
                        else:
                            end_idx = self.idx_dict[next_label[0]] \
                                if cur_label[2:] == next_label[2:] else self.idx_dict[next_label[0]] + 2

                trans_mat[start_idx][end_idx] += 1


class SimilarLengthSampler(Sampler):
    r"""
    Samples elements and ensure
        1. each batch element has similar length to reduce padding.
        2. each batch is in decent length order (useful to pack_sequence for RNN)
        3. batches are ordered randomly
    If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        batch_size (int): num of samples in one batch
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        all_idxs = list(range(len(data_source)))
        all_lens = [self.get_length(idx) for idx in all_idxs]
        self.all_index = self.sort_and_batching(all_idxs, all_lens, batch_size)
        super(SimilarLengthSampler, self).__init__(data_source)

    def sort_and_batching(self, all_idxs, all_lens, batch_size):
        sorted_idxs = sorted(zip(all_idxs, all_lens), key=lambda x: x[1], reverse=True)
        sorted_idxs = [item[0] for item in sorted_idxs]
        batches = self.chunk(sorted_idxs, batch_size)  # shape: (batch_num, batch_size)
        random.shuffle(batches)  # shuffle batches
        flatten_batches = collections._chain.from_iterable(batches)
        return flatten_batches

    def chunk(self, lst, n):
        return [lst[i: i + n] for i in range(0, len(lst), n)]

    def get_length(self, idx):
        return len(self.data_source[idx][0])  # we use the test length in sorting

    def __iter__(self):
        return iter(copy.deepcopy(self.all_index))  # if not deep copy, iteration will stop after first step

    def __len__(self):
        return len(self.data_source)
