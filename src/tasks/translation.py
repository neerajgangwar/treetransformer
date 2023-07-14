import os
import logging
import itertools
import typing as tp
from argparse import Namespace
from functools import partial
from fairseq import utils, search
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from fairseq.data import indexed_dataset, data_utils, Dictionary
from fairseq.tokenizer import tokenize_line
from ..binarizer import VocabularyTreeDatasetBinarizer, TreeFileBinarizer
from ..data import LanguagePairTreeDataset
from ..tree_utils import tree_string_to_symbols
from ..constants import TREE_KEYS
from ..generator import Nstack2SeqGenerator


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl = dataset_impl)

    src_leaves_datasets, src_nodes_datasets, src_spans_datasets, src_postags_datasets, tgt_datasets = [], [], [], [], []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")
        # infer langcode
        if split_exists(split_k, src, tgt, tgt, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, tgt, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError("Dataset not found: {} ({})".format(split, data_path))

        logger.info(f'prefix: {prefix}')

        # src datasets
        src_leaves_dataset = data_utils.load_indexed_dataset(f'{prefix}{src}.leaves', src_dict, dataset_impl)
        src_nodes_dataset = data_utils.load_indexed_dataset(f'{prefix}{src}.nodes', src_dict, dataset_impl)
        src_spans_dataset = data_utils.load_indexed_dataset(f'{prefix}{src}.spans', src_dict, dataset_impl)
        src_postags_dataset = data_utils.load_indexed_dataset(f'{prefix}{src}.pos_tags', src_dict, dataset_impl)
        assert not truncate_source, f'truncate_source is not supported!'
        src_leaves_datasets.append(src_leaves_dataset)
        src_nodes_datasets.append(src_nodes_dataset)
        src_spans_datasets.append(src_spans_dataset)
        src_postags_datasets.append(src_postags_dataset)

        # tgt datasets
        tgt_dataset = data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info("{} {} {}-{} {} examples".format(data_path, split_k, src, tgt, len(src_leaves_datasets[-1])))
        if not combine:
            break

    assert len(src_leaves_datasets) == len(src_nodes_datasets) == len(src_postags_datasets) == len(src_spans_datasets)
    assert len(src_leaves_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_leaves_datasets) == 1:
        src_leaves_dataset = src_leaves_datasets[0]
        src_nodes_dataset = src_nodes_datasets[0]
        src_postags_dataset = src_postags_datasets[0]
        src_spans_dataset = src_spans_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        raise NotImplementedError('Multiple dataset files are not supported!')

    assert not prepend_bos and not prepend_bos_src, f'prepend_bos and prepend_bos_src are not supported for tree_translation'
    assert not append_source_id, f'append_source_id is not supported for tree_translation'
    assert not load_alignments, f'load_alignments is not supported for tree_translation'

    src_dataset_dict = {
        'leaves': src_leaves_dataset,
        'nodes': src_nodes_dataset,
        'pos_tags': src_postags_dataset,
        'spans': src_spans_dataset,
    }

    leave_shape = src_dataset_dict['leaves'].sizes
    node_shape = src_dataset_dict['nodes'].sizes
    src_sizes = leave_shape + node_shape
    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return LanguagePairTreeDataset(
        src_dataset_dict,
        src_sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source = left_pad_source,
        left_pad_target = left_pad_target,
        align_dataset = None,
        eos = None,
        num_buckets = num_buckets,
        shuffle = shuffle,
        pad_to_multiple = pad_to_multiple,
    )


@register_task('tree_translation', dataclass = TranslationConfig)
class TreeTranslationTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super(TreeTranslationTask, self).__init__(cfg, src_dict, tgt_dict)

        # TODO: Getting an error in spans dataset when nmap is used!
        assert cfg.dataset_impl == 'lazy'


    def load_dataset(self, split, epoch = 1, combine = False, **kwargs):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine = combine,
            dataset_impl = self.cfg.dataset_impl,
            upsample_primary = self.cfg.upsample_primary,
            left_pad_source = self.cfg.left_pad_source,
            left_pad_target = self.cfg.left_pad_target,
            max_source_positions = self.cfg.max_source_positions,
            max_target_positions = self.cfg.max_target_positions,
            load_alignments = self.cfg.load_alignments,
            truncate_source = self.cfg.truncate_source,
            num_buckets = self.cfg.num_batch_buckets,
            shuffle = (split != "test"),
            pad_to_multiple = self.cfg.required_seq_len_multiple,
        )


    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        raise NotImplementedError('build_dataset_for_inference is not implemented for tree_transformer!')


    @classmethod
    def build_dictionary(
        cls, filenames, is_srcs, workers=1, threshold=-1, nwords=-1, padding_factor=8,
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            is_srcs (list): list of bool indicating if file at an index is src or tgt
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for idx, filename in enumerate(filenames):
            logger.info(f'Adding to the dictionary: {filename}, is_src: {is_srcs[idx]}')
            Dictionary.add_file_to_dictionary(
                filename,
                d,
                tree_string_to_symbols if is_srcs[idx] else tokenize_line,
                workers,
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d


    @classmethod
    def make_binary_tree_dataset(
        cls,
        vocab: Dictionary,
        input_prefix: str,
        output_prefix: str,
        lang: tp.Optional[str],
        num_workers: int,
        args: Namespace,
    ):
        input_file = '{}{}'.format(input_prefix, ('.' + lang) if lang is not None else '')

        binarizer = VocabularyTreeDatasetBinarizer(
            vocab,
            no_collapse = args.no_collapse,
            reverse_nodes = not args.no_reverse_nodes,
        )
        final_summary = TreeFileBinarizer.multiprocess_dataset(
            input_file,
            args.dataset_impl,
            binarizer,
            output_prefix,
            vocab_size = len(vocab),
            num_workers = num_workers,
        )

        for key in TREE_KEYS:
            logger.info(f'[{lang}] [{key}] {input_file}: {final_summary[key]} (by {vocab.unk_word})')


    # def build_generator(
    #     self,
    #     models,
    #     args,
    #     seq_gen_cls=None,
    #     extra_gen_cls_kwargs=None,
    #     prefix_allowed_tokens_fn=None,
    # ):
    #     assert seq_gen_cls is None, f'seq_gen_cls must be None but got {seq_gen_cls}.'
    #     return super().build_generator(
    #         models,
    #         args,
    #         Nstack2SeqGenerator,
    #         extra_gen_cls_kwargs,
    #         prefix_allowed_tokens_fn,
    #     )
