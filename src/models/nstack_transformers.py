import torch.nn as nn
from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoderModel,
)
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    Embedding,
    Linear,
)
from ..modules.nstack_transformer_layers import (
    NstackMergeTransformerEncoder,
    NstackMerge2SeqTransformerDecoder,
)
from ..modules.nstack_merge_tree_attention import WeightMask, MergeStackNodesOnValueAttention


def build_transformer_embedding(args, dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()

    if path:
        raise NotImplementedError('Not implemented!')
    else:
        print(f'Build new random Embeddings...')
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
    # print(f'Embedding stats: max={emb.weight.max()} - min={emb.weight.min()} - mean={emb.weight.mean()}')
    return emb



@register_model('nstack_merge2seq')
class NstackMerge2SeqTransformer(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--use_pos', default=False, action='store_true')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        assert not args.left_pad_source, f'args.left_pad_source = {args.left_pad_source}, should not be True'
        args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
        args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = args.decoder_type(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, prev_output_tokens, **kwargs):
        try:
            encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
            assert encoder_output is not None, f'encoder_out is None!'
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_output, **kwargs)
        except RuntimeError as er:
            if 'out of memory' in str(er):
                ls = src_node_leaves.size()
                ns = src_node_nodes.size()
                print(f'| WARNING-FORWARD: [{ls},{ns}] OOM exception: {str(er)};\n Skipping batch')
            raise er
        return decoder_out


def nstack_class_base(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.placeholder_const = getattr(args, 'placeholder_const', False)
    args.pretrain_embed_mode = getattr(args, 'pretrain_embed_mode', 'const')
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)

    args.src_len_norm = getattr(args, 'src_len_norm', 'none')
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', False)

    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', True)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'none')

    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.DEFAULT)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', None)

    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)

    args.take_full_dim = getattr(args, 'take_full_dim', False)
    args.hier_embed_right = getattr(args, 'hier_embed_right', False)

    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'none')
    args.node_embed_init = getattr(args, 'node_embed_init', 'embed')

    args.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', False)

    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', False)

    args.vanilla_layers = getattr(args, 'vanilla_layers', 0)

    args.transition_act = getattr(args, 'transition_act', 'none')
    args.transition_dropout = getattr(args, 'transition_dropout', 0.0)

    args.mutual_ancestor_level = getattr(args, 'mutual_ancestor_level', 5)
    args.sep_dwstack_proj_act = getattr(args, 'sep_dwstack_proj_act', 'tanh')

    args.nstack_cross = getattr(args, 'nstack_cross', True)

    args.input_dropout = getattr(args, 'input_dropout', 0)
    base_architecture(args)


def add_iwslt(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)


# @register_model_architecture('nstack2seq', 'nstack2seq_base')
def nstack2seq_base(args):
    nstack_class_base(args)
