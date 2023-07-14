from fairseq.models import register_model_architecture
from .nstack_transformers import nstack_class_base
from ..modules.nstack_transformer_layers import (
    NstackMergeTransformerEncoder,
    NstackMerge2SeqTransformerDecoder,
)
from ..modules.nstack_merge_tree_attention import MergeStackNodesOnValueAttention, WeightMask
from .nstack_transformers import add_iwslt, nstack2seq_base


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    # add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross')
def dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    # args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    # add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)
