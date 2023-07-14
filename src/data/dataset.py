import torch
import numpy as np
from fairseq.data import FairseqDataset, data_utils


def collate_spans(
    values,
    pad_idx,
    pad_to_length = None,
    pad_to_multiple = 1,
    pad_to_bsz = None,
):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size, 2).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][: len(v)])

    return res


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source = True,
    left_pad_target = False,
    input_feeding = True,
    pad_to_length = None,
    pad_to_multiple = 1,
):
    assert not left_pad_source

    if len(samples) == 0:
        return {}

    def merge_source(left_pad, pad_to_length = None):
        assert samples[0]['source'] is not None
        assert not left_pad

        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

        s_leaves = data_utils.collate_tokens(
            src['leaves'],
            pad_idx,
            eos_idx,
            left_pad,
            False,
            pad_to_length = pad_to_length,
            pad_to_multiple = pad_to_multiple,
        )
        s_pos_tags = data_utils.collate_tokens(
            src['pos_tags'],
            pad_idx,
            eos_idx,
            left_pad,
            False,
            pad_to_length = pad_to_length,
            pad_to_multiple = pad_to_multiple,
        )
        s_nodes = data_utils.collate_tokens(
            src['nodes'],
            pad_idx,
            eos_idx,
            left_pad,
            False,
            pad_to_length = pad_to_length,
            pad_to_multiple = pad_to_multiple,
        )
        s_spans = collate_spans(
            src['spans'],
            pad_idx,
            pad_to_length,
            pad_to_multiple,
        )
        return {
            'node_leaves': s_leaves,
            'node_nodes': s_nodes,
            'label_leaves': s_pos_tags,
            'label_nodes': s_nodes,
            'node_indices': s_spans,
        }


    def merge(key, left_pad, move_eos_to_beginning = False, pad_to_length = None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length = pad_to_length,
            pad_to_multiple = pad_to_multiple,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge_source(
        left_pad = left_pad_source,
        pad_to_length = pad_to_length['source'] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s['source']['nodes'].ne(pad_idx).long().sum() + s['source']['leaves'].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = {k : v.index_select(0, sort_order) for k, v in src_tokens.items()}

    src_tokens_ln = torch.cat([src_tokens['node_leaves'], src_tokens['node_nodes']], 1)
    # src_labels = torch.cat([src_tokens['label_leaves'], src_tokens['label_nodes']], 1)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge(
            'target',
            left_pad = left_pad_target,
            pad_to_length = pad_to_length['target']
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s['target'].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get('prev_output_tokens', None) is not None:
            prev_output_tokens = merge('prev_output_tokens', left_pad = left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad = left_pad_target,
                move_eos_to_beginning = True,
                pad_to_length = pad_to_length['target']
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_node_leaves': src_tokens['node_leaves'],
            'src_node_nodes': src_tokens['node_nodes'],
            'src_label_leaves': src_tokens['label_leaves'],
            'src_label_nodes': src_tokens['label_nodes'],
            'src_node_indices': src_tokens['node_indices'],
            'src_lengths': src_lengths,
            'src_tokens': src_tokens_ln,
            # 'src_labels': src_labels,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens.index_select(0, sort_order)

    if samples[0].get('alignment', None) is not None:
        raise NotImplementedError('alignment is not implemented!')

    if samples[0].get('constraints', None) is not None:
        raise NotImplementedError('constraints is not implemented!')

    return batch


class LanguagePairTreeDataset(FairseqDataset):
    def __init__(
        self,
        srcs,
        src_sizes,
        src_dict,
        tgt = None,
        tgt_sizes = None,
        tgt_dict = None,
        left_pad_source = True,
        left_pad_target = False,
        shuffle = True,
        input_feeding = True,
        remove_eos_from_source = False,
        append_eos_to_target = False,
        align_dataset = None,
        constraints = None,
        append_bos = False,
        eos = None,
        num_buckets = 0,
        src_lang_id = None,
        tgt_lang_id = None,
        pad_to_multiple = 1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()

        self.srcs = srcs
        self.src_leaves = srcs['leaves']
        self.src_nodes = srcs['nodes']
        self.src_pos_tags = srcs['pos_tags']
        self.src_spans = srcs['spans']
        self.tgt = tgt
        assert len(self.src_leaves) == len(self.src_nodes) == len(self.src_pos_tags) == len(self.src_spans)
        if tgt is not None:
            assert len(self.src_leaves) == len(self.tgt), 'Source and target must contain the same number of examples'

        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, 'Both source and target needed when alignments are provided'
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            raise NotImplementedError('num_buckets > 0 is not supported!')
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple


    def get_batch_shapes(self):
        return self.buckets


    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = {k: v[index] for k, v in self.srcs.items()}

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa.
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            assert tgt_item.numel() > 0
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            raise NotImplementedError('append_bos is not implemented!')

        if self.remove_eos_from_source:
            raise NotImplementedError('remove_eos_from_source is not implemented!')

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

        if self.align_dataset is not None:
            raise NotImplementedError('align_dataset is not implemented!')

        if self.constraints is not None:
            raise NotImplementedError('constraints is not implemented!')

        return example


    def __len__(self):
        return len(self.src_leaves)


    def collater(self, samples, pad_to_length = None):
        res = collate(
            samples = samples,
            pad_idx = self.src_dict.pad(),
            eos_idx = self.src_dict.eos(),
            left_pad_source = self.left_pad_source,
            left_pad_target = self.left_pad_target,
            input_feeding = self.input_feeding,
            pad_to_length = pad_to_length,
            pad_to_multiple = self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res['net_input']['src_tokens']
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res['net_input']['src_lang_id'] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res['tgt_lang_id'] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res


    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0
        )


    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes


    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0
        )


    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        if self.buckets is None:
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind = 'mergesort')]
            return indices[np.argsort(self.src_sizes[indices], kind = 'mergesort')]
        else:
            raise NotImplementedError('buckets is not implemented!')


    def prefetch(self, indices):
        for k, v in self.srcs.items():
            v.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            raise NotImplementedError('align_dataset is not implemented!')


    @property
    def supports_prefetch(self):
        return getattr(self.src_leaves, "supports_prefetch", False) and \
            getattr(self.src_nodes, "supports_prefetch", False) and \
            getattr(self.src_pos_tags, "supports_prefetch", False) and \
            getattr(self.src_spans, "supports_prefetch", False) and \
            (getattr(self.tgt, "supports_prefetch", False) or self.tgt is None)


    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )
