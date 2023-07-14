import torch
import logging
from collections import Counter
from multiprocessing import Pool
from ..tree_utils import tree_string_to_leave_pos_node_span


logger = logging.getLogger(__name__)


class TreeTokenizer:
    @staticmethod
    def tokenize(
        words,
        vocab,
        add_if_not_exist = True,
        consumer = None,
        append_eos = True,
        reverse_order = False,
    ):
        if reverse_order:
            words = list(reversed(words))

        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = vocab.add_symbol(word)
            else:
                idx = vocab.index(word)

            if consumer is not None:
                consumer(word, idx)

            ids[i] = idx

        if append_eos:
            ids[nwords] = vocab.eos_index

        return ids


    @staticmethod
    def line2example(
        s,
        vocab,
        consumer,
        append_eos = False,
        reverse_order = False,
        add_if_not_exist = False,
        no_collapse = False,
        label_only = False,
        tolower = False,
    ):
        leaves, pos_tags, nodes, spans = tree_string_to_leave_pos_node_span(s, no_collapse = no_collapse)

        if tolower:
            leaves = ' '.join(leaves).lower().split()
            pos_tags = ' '.join(pos_tags).lower().split()
            nodes = ' '.join(nodes).lower().split()

        leave_indices = TreeTokenizer.tokenize(
            words = leaves,
            vocab = vocab,
            add_if_not_exist = add_if_not_exist,
            consumer = consumer,
            append_eos = append_eos,
            reverse_order = reverse_order,
        )
        if label_only:
            pos_tag_indices = torch.tensor([int(x) for x in pos_tags]).int()
            node_indices = torch.tensor([int(x) for x in nodes]).int()
        else:
            pos_tag_indices = TreeTokenizer.tokenize(
                words = pos_tags,
                vocab = vocab,
                add_if_not_exist = add_if_not_exist,
                consumer = consumer,
                append_eos = append_eos,
                reverse_order = reverse_order,
            )

            node_indices = TreeTokenizer.tokenize(
                words = nodes,
                vocab = vocab,
                add_if_not_exist = add_if_not_exist,
                consumer = consumer,
                append_eos = append_eos,
                reverse_order = reverse_order,
            )

        span_indices = torch.tensor(spans).int()
        assert span_indices.dim() == 2, f'{s}: {leaves}, {pos_tags}, {nodes}, {spans}'
        assert span_indices.size(0) == node_indices.size(0)

        example = {
            'leaves': leave_indices,
            'nodes': node_indices,
            'pos_tags': pos_tag_indices,
            'spans': span_indices
        }
        return example
