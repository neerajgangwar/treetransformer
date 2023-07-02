import os
import torch
import logging
import typing as tp
from collections import Counter
from functools import partial
from multiprocessing import Pool
from fairseq.data import Dictionary, indexed_dataset
from fairseq.binarizer import BinarizeSummary, _worker_prefix
from fairseq.file_chunker_utils import Chunker, find_offsets
from fairseq.file_io import PathManager
from .tree_utils import tree_string_to_leave_pos_node_span
from .constants import TREE_KEYS


logger = logging.getLogger(__name__)


class VocabularyTreeDatasetBinarizer:
    def __init__(
        self,
        dict: Dictionary,
        no_collapse: bool,
        reverse_nodes: bool,
    ) -> None:
        self.dict = dict
        self.no_collapse = no_collapse
        self.reverse_nodes = reverse_nodes


    def binarize_line(
        self,
        line: str,
        leaves_summary: BinarizeSummary,
        nodes_summary: BinarizeSummary,
        pos_tags_summary: BinarizeSummary,
        spans_summary: BinarizeSummary,
    ):
        if leaves_summary.replaced is None:
            leaves_summary.replaced = Counter()

        if nodes_summary.replaced is None:
            nodes_summary.replaced = Counter()

        if pos_tags_summary.replaced is None:
            pos_tags_summary.replaced = Counter()

        def replaced_consumer(word, idx, modality):
            if idx == self.dict.unk_index and word != self.dict.unk_word:
                if modality == 'leaves':
                    leaves_summary.replaced.update([word])
                elif modality == 'nodes':
                    nodes_summary.replaced.update([word])
                elif modality == 'pos_tags':
                    nodes_summary.replaced.update([word])
                else:
                    raise NotImplementedError(f'binarize_line.replaced_consumer is not implemented for modality "{modality}"')

        leaves, pos_tags, nodes, spans = tree_string_to_leave_pos_node_span(line, self.no_collapse)
        leaves_ids = self.dict.encode_line(
            line = leaves,
            line_tokenizer = lambda x: x,
            add_if_not_exist = False,
            consumer = partial(replaced_consumer, modality = 'leaves'),
            append_eos = False,
            reverse_order = False,
        )
        pos_tags_ids = self.dict.encode_line(
            line = pos_tags,
            line_tokenizer = lambda x: x,
            add_if_not_exist = False,
            consumer = partial(replaced_consumer, modality = 'pos_tags'),
            append_eos = False,
            reverse_order = False,
        )
        nodes_ids = self.dict.encode_line(
            line = nodes,
            line_tokenizer = lambda x: x,
            add_if_not_exist = False,
            consumer = partial(replaced_consumer, modality = 'nodes'),
            append_eos = False,
            reverse_order = False,
        )
        spans_ids = torch.IntTensor(spans)

        if self.reverse_nodes:
            nodes_ids = nodes_ids.flip([0])
            spans_ids = spans_ids.flip([0])

        leaves_summary.num_seq += 1
        leaves_summary.num_tok += len(leaves_ids)
        pos_tags_summary.num_seq += 1
        pos_tags_summary.num_tok += len(pos_tags_ids)
        nodes_summary.num_seq += 1
        nodes_summary.num_tok += len(nodes_ids)
        spans_summary.num_seq += 1

        return leaves_ids, pos_tags_ids, nodes_ids, spans_ids


class TreeFileBinarizer:
    """
    An file binarizer can take a file, tokenize it, and binarize each line to a tensor
    """

    @classmethod
    def multiprocess_dataset(
        cls,
        input_file: str,
        dataset_impl: str,
        binarizer: VocabularyTreeDatasetBinarizer,
        output_prefix: str,
        vocab_size=None,
        num_workers=1,
    ) -> BinarizeSummary:
        final_summary = {key: BinarizeSummary() for key in TREE_KEYS}

        offsets = find_offsets(input_file, num_workers)
        # find_offsets returns a list of position [pos1, pos2, pos3, pos4] but we would want pairs:
        # [(pos1, pos2), (pos2, pos3), (pos3, pos4)] to process the chunks with start/end info
        # we zip the list with itself shifted by one to get all the pairs.
        (first_chunk, *more_chunks) = zip(offsets, offsets[1:])
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            worker_results = [
                pool.apply_async(
                    cls._binarize_chunk_and_finalize,
                    args=(
                        binarizer,
                        input_file,
                        start_offset,
                        end_offset,
                        _worker_prefix(
                            output_prefix,
                            worker_id,
                        ),
                        dataset_impl,
                    ),
                    kwds={
                        "vocab_size": vocab_size,
                    }
                    if vocab_size is not None
                    else {},
                )
                for worker_id, (start_offset, end_offset) in enumerate(
                    more_chunks, start=1
                )
            ]

            pool.close()
            pool.join()
            for r in worker_results:
                summ = r.get()
                for key in TREE_KEYS:
                    final_summary[key].merge(summ[key])

        # do not close the bin file as we need to merge the worker results in
        final_ds, summ = cls._binarize_file_chunk(
            binarizer,
            input_file,
            offset_start=first_chunk[0],
            offset_end=first_chunk[1],
            output_prefix=output_prefix,
            dataset_impl=dataset_impl,
            vocab_size=vocab_size if vocab_size is not None else None,
        )
        for key in TREE_KEYS:
            final_summary[key].merge(summ[key])

        if num_workers > 1:
            for worker_id in range(1, num_workers):
                # merge the worker outputs
                worker_output_prefix = _worker_prefix(
                    output_prefix,
                    worker_id,
                )
                for key in TREE_KEYS:
                    worker_output_prefix_key = f'{worker_output_prefix}.{key}'
                    final_ds[key].merge_file_(worker_output_prefix_key)
                    try:
                        os.remove(indexed_dataset.data_file_path(worker_output_prefix_key))
                        os.remove(indexed_dataset.index_file_path(worker_output_prefix_key))
                    except Exception as e:
                        logger.error(
                            f"couldn't remove {worker_output_prefix_key}.*", exc_info=e
                        )

        #  now we can close the file
        for key in TREE_KEYS:
            idx_file = indexed_dataset.index_file_path(f'{output_prefix}.{key}')
            final_ds[key].finalize(idx_file)

        return final_summary

    @staticmethod
    def _binarize_file_chunk(
        binarizer: VocabularyTreeDatasetBinarizer,
        filename: str,
        offset_start: int,
        offset_end: int,
        output_prefix: str,
        dataset_impl: str,
        vocab_size=None,
    ) -> tp.Tuple[tp.Any, tp.List[BinarizeSummary]]:  # (dataset builder, BinarizeSummary)
        """
        creates a dataset builder and append binarized items to it. This function does not
        finalize the builder, this is useful if you want to do other things with your bin file
        like appending/merging other files
        """
        ds = {}
        summary = {}
        for key in TREE_KEYS:
            bin_file = indexed_dataset.data_file_path(f'{output_prefix}.{key}')
            ds[key] = indexed_dataset.make_builder(
                bin_file,
                impl = dataset_impl,
                vocab_size = vocab_size,
            )
            summary[key] = BinarizeSummary()

        with Chunker(
            PathManager.get_local_path(filename), offset_start, offset_end
        ) as line_iterator:
            for line in line_iterator:
                leaves_ids, pos_tags_ids, nodes_ids, spans_ids = binarizer.binarize_line(
                    line,
                    summary['leaves'],
                    summary['nodes'],
                    summary['pos_tags'],
                    summary['spans'],
                )
                ds['leaves'].add_item(leaves_ids)
                ds['pos_tags'].add_item(pos_tags_ids)
                ds['nodes'].add_item(nodes_ids)
                ds['spans'].add_item(spans_ids)

        return ds, summary

    @classmethod
    def _binarize_chunk_and_finalize(
        cls,
        binarizer: VocabularyTreeDatasetBinarizer,
        filename: str,
        offset_start: int,
        offset_end: int,
        output_prefix: str,
        dataset_impl: str,
        vocab_size=None,
    ):
        """
        same as above, but also finalizes the builder
        """
        ds, summ = cls._binarize_file_chunk(
            binarizer,
            filename,
            offset_start,
            offset_end,
            output_prefix,
            dataset_impl,
            vocab_size=vocab_size,
        )

        for key in TREE_KEYS:
            full_output_prefix = f'{output_prefix}.{key}'
            idx_file = indexed_dataset.index_file_path(full_output_prefix)
            ds[key].finalize(idx_file)

        return summ

