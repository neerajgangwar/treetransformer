# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from copy import deepcopy
from nltk.parse import CoreNLPParser
import argparse
from nltk.tree import Tree
import os
import traceback
import os
from torch.utils.data import Dataset, DataLoader
from nltk.tree import Tree
from copy import deepcopy
import argparse
from tqdm import tqdm


RETRIEVE_BATCH = int(os.environ.get('RETRIEVE_BATCH', 1000))
PARSER_TIMEOUT = int(os.environ.get('PARSER_TIMEOUT', 60000000))
PARSER_PORT = str(os.environ.get('PARSER_PORT', 9001))

SPECIAL_CHAR = {'&apos;': "'", '&apos;s': "'s", '&quot;': '"', '&#91;': '[',
                '&#93;': "]", '&apos;@@': "'@@", '&apos;t': "'t",
                '&amp;': "&", '&apos;ll': "'ll", '&apos;ve': "'ve",
                '&apos;m': "'m", '&apos;re': "'re", '&apos;d': "'d",
                '&#124;': "|", '&gt;': ">", '&lt;': "<"}

# special_character_dict['&quot;']="''"
SPECIAL_CHAR_MBACK = {v: k for k, v in SPECIAL_CHAR.items()}
SPECIAL_CHAR_MBACK['-LSB-'] = '&#91;'
SPECIAL_CHAR_MBACK['-RSB-'] = '&#93;'
SPECIAL_CHAR_MBACK['-LRB-'] = "("
SPECIAL_CHAR_MBACK['-RRB-'] = ")"
SPECIAL_CHAR_MBACK["''"] = "&quot;"


class CustomCoreNLPParser(CoreNLPParser):
    def api_call(self, data, properties=None, timeout=18000000, lang=None):
        if properties is None:
            properties = {"parse.binaryTrees": "true"}
        return super().api_call(data, properties, timeout)


    @classmethod
    def build_parser(self, host: str, port: int):
        return self(url = f"http://{host}:{port}")


def tree_str_post_process(tree_string):
    tree_string = tree_string.replace('-LRB- (', '-LRB- -LRB-').replace('-RRB- )', '-RRB- -RRB-')
    tree_string = tree_string.replace('TRUNC (', 'TRUNC -LRB-').replace('TRUNC )', 'TRUNC -RRB-')
    return tree_string


def tree_from_string(tree_string):
    try:
        s = tree_string
        s = tree_str_post_process(s)
        tree = Tree.fromstring(s)
    except Exception as e:
        try:
            tree = Tree.fromstring(tree_string)
        except Exception as e:
            print(f'ERROR: unable to parse the tree')
            print(tree_string)
            raise e
    return tree


# TODO: Nstack Dataset builder ---------------------
def replace_special_character(string):
    new_string = deepcopy(string)
    new_string = new_string.replace(")", u"）").replace("(", u"（")

    list_string = new_string.split(" ")
    new_list = deepcopy(list_string)
    for i in range(len(list_string)):
        for k, v in SPECIAL_CHAR.items():
            if k in list_string[i]:
                new_list[i] = list_string[i].replace(k, v)
    return " ".join(new_list)


def merge_list_tree(list_tree):
    root_label = [x.label() for x in list_tree]
    assert len(set(root_label)) == 1 and 'ROOT' in root_label
    list_string = "(ROOT " + " ".join([str(i) for i in range(len(list_tree))]) + ")"
    new_tree = Tree.fromstring(list_string)
    for i in range(len(list_tree)):
        new_tree[i] = list_tree[i][0]
    return new_tree


def remap_chars(tree):
    for i in range(len(tree.leaves())):
        if tree.leaves()[i] in SPECIAL_CHAR_MBACK:
            tree[tree.leaf_treeposition(i)] = SPECIAL_CHAR_MBACK[tree.leaves()[i]]


def parse_string(parser, bpe_string):
    word_string_nobpe = bpe_string.replace("@@ ", "")
    word_string = replace_special_character(word_string_nobpe)
    try:
        tree_strings = list(parser.parse_text(word_string))
    except Exception as e:
        try:
            print(f'Try bpe version')
            tree_strings = list(parser.parse_text(bpe_string))
        except Exception as ee:
            print(f'Failed.')
            print(f'[Ori]: {bpe_string}')
            print(f'[Proc]: {word_string}')
            traceback.print_stack()
            raise ee

    merged = merge_list_tree(tree_strings)
    remap_chars(merged)
    parse_string = ' '.join(str(merged).split())
    token_set = set(merged.leaves())
    return parse_string, token_set


class Nstack2SeqDataset(Dataset):
    def __init__(self, src_file: str, tgt_file: str, parser_host: str, parser_port: int):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.src_data = open(src_file, "r").readlines()
        self.tgt_data = open(tgt_file, "r").readlines()
        assert len(self.src_data) == len(self.tgt_data), f'Incosistent length {len(self.src_data)} != {len(self.tgt_data)}'

        self.parser_host = parser_host
        self.parser_port = parser_port
        self.parser = CustomCoreNLPParser.build_parser(host = parser_host, port = parser_port)


    def __len__(self):
        return len(self.src_data)


    def __getitem__(self, idx):
        src_sample = self.src_data[idx].rstrip("\n")
        tgt_sample = self.tgt_data[idx].rstrip("\n")
        try:
            parse_strings, token_string_set = parse_string(self.parser, src_sample)
        except Exception as e:
            print(f'Error happen at index {idx}, return empty')
            print(f'{src_sample}')
            print(f'{tgt_sample}')
            print(traceback.format_exc())
            raise e

        src_sample = {
            'ori_sample': src_sample,
            'pstring': parse_strings,
            'token_set': list(token_string_set),
            'tgt': tgt_sample,
        }
        return src_sample


class TreeBuilder(object):
    SENT_SPLITTER = '#####------#####'

    def __init__(self, transform = None, bpe_code = None, ignore_error = True) -> None:
        super().__init__()
        self.transform = transform
        self.bpe_code = bpe_code
        self.ignore_error = ignore_error

        assert bpe_code is not None and os.path.exists(bpe_code)

        print(f'ignore_error= {self.ignore_error}')
        print(f'bpe_code= {self.bpe_code}')


    def build_bpe_tree_vocab(self, raw_vocab_file_or_set, output_file):
        bpe_vocab_file = f'{output_file}.bpe.vocab'

        if isinstance(raw_vocab_file_or_set, str):
            assert os.path.exists(raw_vocab_file_or_set)
            raw_vocab_file = raw_vocab_file_or_set
        else:
            assert isinstance(raw_vocab_file_or_set, (set, list))
            raw_vocab = list(raw_vocab_file_or_set)
            raw_vocab_file = f"temp.{len(raw_vocab)}.vocab"
            with open(raw_vocab_file, 'w') as f:
                f.write('\n'.join(raw_vocab))

        print(f'Applying BPE: subword-nmt apply-bpe -c {self.bpe_code} < {raw_vocab_file} > {bpe_vocab_file}')
        os.system(f'subword-nmt apply-bpe -c {self.bpe_code} < {raw_vocab_file} > {bpe_vocab_file}')
        assert os.path.exists(bpe_vocab_file)

        # re-open bpe vocab
        with open(bpe_vocab_file, "r") as f:
            vocab_bpe = f.readlines()

        list_dict = [x.strip().replace("@@ ", "") for x in vocab_bpe]
        word2bpe = {}

        for i, w in enumerate(list_dict):
            word2bpe[w] = vocab_bpe[i].strip().split(" ")

        return bpe_vocab_file, word2bpe, list_dict, vocab_bpe


    def export_seq_file(self, data, file, separate = True):
        with open(file, "w") as f:
            for i, w in enumerate(data):
                if separate:
                    assert isinstance(w, (list, tuple))
                    s = self.__class__.SENT_SPLITTER.join(w)
                    f.write(f'{s}\n')
                else:
                    f.write(f'{w}\n')
            f.close()


    def apply_bpe_on_tree_strings(self, data, word2bpe, workers=0):
        parse_string_data = []
        splitter = self.__class__.SENT_SPLITTER

        class BpeToTreeStringDataset(Dataset):
            def __init__(self, data) -> None:
                super().__init__()
                self.data = data


            def __len__(self):
                return len(self.data)


            def __getitem__(self, index):
                data_p = self.data[index]
                data_p = data_p if isinstance(data_p, list) else [data_p]
                bpe_trees = []
                for x in data_p:
                    try:
                        tree_i = tree_from_string(x)
                    except Exception as e:
                        print(f'[{self.__class__}] Error when parse tree at index {index}')
                        raise e

                    new_tree_j = deepcopy(tree_i)
                    leaves = tree_i.leaves()
                    for j, word in enumerate(leaves):
                        if word in word2bpe and len(word2bpe[word]) > 1:
                            # reassign
                            loc_leaf_j = tree_i.leaf_treeposition(j)
                            # eg: (0, 1, 0)...
                            pos_tag_j = tree_i[loc_leaf_j[:-1]].label()
                            bpe_tree_j = Tree(
                                pos_tag_j,
                                [Tree(f'{pos_tag_j}_BPE', [x]) for x in word2bpe[word]]
                            )
                            new_tree_j[loc_leaf_j[:-1]] = bpe_tree_j
                    parsing_bpe_tree = ' '.join(str(new_tree_j).split())
                    bpe_trees.append(parsing_bpe_tree)

                merged = splitter.join(bpe_trees)
                return merged


        dataset = BpeToTreeStringDataset(data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
        for i, d in tqdm(enumerate(dataloader), desc = "BPE Tree"):
            parse_string_data.append(d[0])

        return parse_string_data


    def build_bpe_tree(self, data, raw_vocab_file_or_set, before_bpe_output_file, bpe_output_file):
        print(f'Proceed to generate bpe tree: {bpe_output_file}')
        _, word2bpe, _, _ = self.build_bpe_tree_vocab(raw_vocab_file_or_set, before_bpe_output_file)

        print(f'apply_bpe_on_tree_strings: {bpe_output_file}')
        parse_string_data = self.apply_bpe_on_tree_strings(data, word2bpe)
        with open(bpe_output_file, "w") as f:
            f.write('\n'.join(parse_string_data))
            f.close()


class SeqTreeBuilder(TreeBuilder):
    def __init__(self, transform = None, bpe_code = None, ignore_error = True, parser_host: str = None, parser_port: int = None) -> None:
        super(SeqTreeBuilder, self).__init__(transform, bpe_code, ignore_error)

        self.parser_host = parser_host
        self.parser_port = parser_port


    def retrieve_tree_data(self, dataloader):
        src_data = []
        tgt_data = []
        vocab = set()

        for sample_idx, sample_batched in tqdm(enumerate(dataloader), desc = "Parse Tree"):
            assert isinstance(sample_batched['pstring'], (list, tuple))
            s = sample_batched['pstring'][0]
            assert isinstance(s, str)

            v = [ts[0] for ts in sample_batched['token_set']]
            tgt = sample_batched['tgt'][0]

            src_data.append(s)
            tgt_data.append(tgt)
            vocab = vocab.union(set(v))

        vocab = list(vocab)
        return src_data, tgt_data, vocab


    def export_text_to_tree_strings(self, input_file, output_file, num_workers = 0, tgt_file = None, tgt_out_file = None):
        assert tgt_file is not None and os.path.exists(tgt_file)
        assert tgt_out_file is not None and tgt_out_file != ''

        raw_vocab_file = f'{output_file}.raw.vocab'
        before_bpe_output_file = f'{output_file}.before-bpe'
        print(f'Processing file {input_file}')
        print(f'Output to {output_file}')

        dataset = Nstack2SeqDataset(input_file, tgt_file, self.parser_host, self.parser_port)
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = num_workers)
        src_data, tgt_data, vocab = self.retrieve_tree_data(dataloader)

        print(f'Generating raw vocab: {raw_vocab_file}')
        self.export_seq_file(vocab, raw_vocab_file, separate = False)
        print(f'Generate tree string data: {output_file}')
        self.export_seq_file(src_data, before_bpe_output_file, separate = False)
        print(f'Generate tgt data: {tgt_out_file}')
        self.export_seq_file(tgt_data, tgt_out_file, separate = False)

        print(f'Start building BPE tree')
        self.build_bpe_tree(src_data, vocab, before_bpe_output_file, output_file)




"""
export split=train
export before=/Users/xnguyen/Desktop/projects/nmt/raw_data/iwslt14.tokenized.de-en/$split.bpe.sep.en.before-bpe
export after=/Users/xnguyen/Desktop/projects/nmt/raw_data/iwslt14.tokenized.de-en/$split.bpe.sep.en.after-bpe
export vocab=/Users/xnguyen/Desktop/projects/nmt/raw_data/iwslt14.tokenized.de-en/$split.bpe-1.unify-0.en.raw.vocab
export code=/Users/xnguyen/Desktop/projects/nmt/raw_data/iwslt14.tokenized.de-en/code

export split=valid
export before=/Users/xnguyen/Desktop/projects/nmt/raw_data/SST-2-glue-nstack-sep-bpe/$split.input
export after=/Users/xnguyen/Desktop/projects/nmt/raw_data/SST-2-glue-nstack-sep-bpe/$split.input.bpe
export vocab=/Users/xnguyen/Desktop/projects/nmt/raw_data/SST-2-glue-nstack-sep-bpe/$split.input.raw.vocab
export code=/Users/xnguyen/Desktop/projects/nmt/raw_data/SST-2-glue-nstack-sep-bpe/bpe.32k

export split=valid
export split=test
export before=/Users/xnguyen/Desktop/projects/nmt/raw_data/sva_number_prediction-nstack-sep-bpe/$split.input
export after=/Users/xnguyen/Desktop/projects/nmt/raw_data/sva_number_prediction-nstack-sep-bpe/$split.input.bpe
export vocab=/Users/xnguyen/Desktop/projects/nmt/raw_data/sva_number_prediction-nstack-sep-bpe/$split.input.raw.vocab
export code=/Users/xnguyen/Desktop/projects/nmt/raw_data/sva_number_prediction-nstack-sep-bpe/bpe.32k


export split=train
export root_dir=/Users/xnguyen/Desktop/projects/nmt/raw_data/SST-2-glue-nstack-sep-bpe
export before=$root_dir/$split.input
export after=$root_dir/$split.bpe4k.input
export vocab=$root_dir/$split.input.raw.vocab
export code=$root_dir/bpe.4k
cp -r $root_dir/$split.target $root_dir/$split.bpe4k.target
python nstack_process.py --before $before --after $after --raw_vocab $vocab --bpe_code $code


export split=test
export root_dir=/Users/xnguyen/Desktop/projects/nmt/raw_data/sva_number_prediction-nstack-sep-bpe
export before=$root_dir/$split.input
export after=$root_dir/$split.bpe4k.input
export vocab=$root_dir/$split.input.raw.vocab
export code=$root_dir/bpe.4k
cp -r $root_dir/$split.target $root_dir/$split.bpe4k.target
python nstack_process.py --before $before --after $after --raw_vocab $vocab --bpe_code $code



# BERT

export split=test
export bert_name="bert-base-uncased"
export bert_cache=/Users/xnguyen/Desktop/projects/nmt/pt_bert

export root_dir=/Users/xnguyen/Desktop/projects/nmt/raw_data/SST-2-glue-nstack-sep-bpe
export root_dir=/Users/xnguyen/Desktop/projects/nmt/raw_data/sva_number_prediction-nstack-sep-bpe

export before=$root_dir/$split.input
export after=$root_dir/$split.$bert_name.input
export vocab=$root_dir/$split.input.raw.vocab
export code=$root_dir/bpe.4k

cp -r $root_dir/$split.target $root_dir/$split.$bert_name.target
python nstack_process.py --before $before --after $after --raw_vocab $vocab --bpe_code $code --bert_name $bert_name --bert_cache_dir $bert_cache



# COLA
export root_dir=/Users/xnguyen/Desktop/projects/nmt/raw_data/cola/tmp_cola/CoLA
export split=dev
export before=$root_dir/$split.input
export after=$root_dir/$split.tree.input

cp -r $root_dir/$split.target $root_dir/$split.tree.target

python nstack_process.py --before $before --after $after
"""

# todo: new process.....
"""
export split=train
export before=/Users/xnguyen/Desktop/projects/nmt/raw_data/iwslt14.tokenized.de-en.v2/$split.bpe.sep.en.before-bpe
export after=/Users/xnguyen/Desktop/projects/nmt/raw_data/iwslt14.tokenized.de-en.v2/$split.bpe.sep.en.after-bpe
export vocab=""
export code=/Users/xnguyen/Desktop/projects/nmt/raw_data/iwslt14.tokenized.de-en.v2/code
python nstack_process.py --before $before --after $after --bpe_code $code



English-Vietnamese

export prefix=train.lower
# export prefix=valid.lower
# export prefix=test.lower

export cur=`pwd`
export PARSER_PORT=9001
export root=${cur}/../../../raw_data/translate_envi
export before=$root/$prefix.en
export after=$root/$prefix.tree.en
export log_file=$root/parse.${prefix}.log
# turn off ignore error with valid/test
python -u nstack_process.py --ignore_error --before $before --after $after --before_tgt ${root}/${prefix}.vi --after_tgt ${root}/${prefix}.tree.vi 2>&1 | tee ${log_file}
# python -u nstack_process.py --before $before --after $after --before_tgt ${root}/${prefix}.vi --after_tgt ${root}/${prefix}.tree.vi 2>&1 | tee ${log_file}


English-Romanian

export prefix=train.lower
# valid.tok.lower
export prefix=train.tok.clean.lower
export PARSER_PORT=9002
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_enro
export before=$root/$prefix.en
export after=$root/$prefix.tree.en
export log_file=$root/parse.${prefix}.log
# python -u nstack_process.py --before $before --after $after 2>&1 | tee ${log_file}
python -u nstack_process.py --ignore_error --before $before --after $after --before_tgt ${root}/${prefix}.ro --after_tgt ${root}/${prefix}.tree.ro 2>&1 | tee ${log_file}


# with BPE, need to parse
IWSLT English-French

export prefix=train.tok.clean.lower.bpe32768
export prefix=valid.bpe32768
export prefix=test.bpe32768
export PARSER_PORT=9004
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt_enfr
export tgt=fr
export src=en
export after_suffix=tree.bpe
export before=$root/$prefix.${src}
export after=$root/$prefix.${after_suffix}.${src}
export before_tgt=${root}/${prefix}.${tgt}
export after_tgt=${root}/$prefix.${after_suffix}.${tgt}

export bpe=${root}/bpe.32768
export log_file=$root/parse.${prefix}.${after_suffix}.log
python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --convert_bpe --before $before --after $after --before_tgt ${before_tgt} --after_tgt ${after_tgt} 2>&1 | tee ${log_file}
# python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${root}/${prefix}.fr --after_tgt ${root}/${prefix}.tree.fr 2>&1 | tee ${log_file}

# IWSLT - EN-FR convert to bpe

export prefix=test.bpe32768
export prefix=valid.bpe32768
export prefix=train.tok.clean.lower.bpe32768
export PARSER_PORT=9004
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt_enfr
export tgt=fr
export src=en
export after_suffix=tree.bpe
export before=$root/$prefix.tree.${src}
export after=$root/$prefix.${after_suffix}.${src}
export bpe=${root}/bpe.32768
export log_file=$root/parse.${prefix}.${after_suffix}.convert.log
cp -r $root/$prefix.tree.${tgt} $root/$prefix.${after_suffix}.${tgt}
python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --convert_bpe --before $before --after $after  2>&1 | tee ${log_file}
wc -l $root/$prefix.${after_suffix}.${tgt}
wc -l $after



WMT'16 English-German
# newstest2013.tok.bpe.32000.en newstest2013.tok.bpe.32000.de
# newstest2014.tok.bpe.32000.en newstest2014.tok.bpe.32000.de
# wmt16_ende
# bpe.32000			newstest2013.tok.bpe.32000.de	newstest2013.tok.bpe.32000.en	newstest2014.tok.bpe.32000.de
# newstest2014.tok.bpe.32000.en	train.tok.clean.bpe.32000.de	train.tok.clean.bpe.32000.en	vocab.bpe.32000


export prefix=train.tok.clean.bpe.32000
export prefix=newstest2014.tok.bpe.32000
export prefix=newstest2013.tok.bpe.32000

export cur=`pwd`
export root=${cur}/../../../raw_data/wmt16_ende
export PARSER_PORT=9004
export before=$root/$prefix.en
export after=$root/$prefix.tree.en
export bpe=${root}/bpe.32000
export log_file=$root/parse.log
# cp -r ${root}/${prefix}.de ${root}/${prefix}.tree.de
# python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after 2>&1 | tee ${log_file}
python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${root}/${prefix}.de --after_tgt ${root}/${prefix}.tree.de 2>&1 | tee ${log_file}
# python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${root}/${prefix}.de --after_tgt ${root}/${prefix}.tree.de 2>&1 | tee ${log_file}


WMT'16 English-German -> convert to BPE
export prefix=newstest2014.tok.bpe.32000.tree
export prefix=newstest2013.tok.bpe.32000.tree
export prefix=train.tok.clean.bpe.32000.combined.tree

export cur=`pwd`
export root=${cur}/../../../raw_data/wmt16_ende
export PARSER_PORT=9000
export before=$root/$prefix.en
export after=$root/$prefix.bpetree.en
export before_tgt=$root/$prefix.de
export after_tgt=${root}/${prefix}.bpetree.de
export bpe=${root}/bpe.32000
export log_file=$root/parse.tobpe-${prefix}.log
python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_tgt} --after_tgt ${after_tgt} 2>&1 | tee ${log_file}


IWSLT'14 En-De -> convert to BPE, other version
test.nobpe.p135.flat.de  test.nobpe.p330.flat.de  train.nobpe.p135.flat.de  train.nobpe.p330.flat.de  valid.nobpe.p135.flat.de  valid.nobpe.p330.flat.de
test.nobpe.p135.flat.en  test.nobpe.p330.flat.en  train.nobpe.p135.flat.en  train.nobpe.p330.flat.en  valid.nobpe.p135.flat.en  valid.nobpe.p330.flat.en

export prefix=test.nobpe.p135.flat
export prefix=test.nobpe.p330.flat
export prefix=valid.nobpe.p135.flat
export prefix=valid.nobpe.p330.flat
export prefix=train.nobpe.p135.flat
export prefix=train.nobpe.p330.flat

export cur=`pwd`
export root=${cur}/../../../raw_data/iwslt14.tokenized.de-en.v2
export PARSER_PORT=9000
export before=$root/$prefix.en
export after=$root/$prefix.bpetree.en
export before_tgt=$root/$prefix.de
export after_tgt=${root}/${prefix}.bpetree.de
export bpe=${root}/code
export log_file=$root/parse.tobpe-${prefix}.log
python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_tgt} --after_tgt ${after_tgt} 2>&1 | tee ${log_file}


# --------------------------------------------------------------------
# todo: WMT'14 ENglish French
# /export/home/projects/nmt/raw_data/wmt14_en_fr/train.p24.en

export prefix=test
export cur=`pwd`
export root=${cur}/../../../raw_data/wmt14_en_fr
export before=$root/$prefix.en
export after=$root/$prefix.tree.en
export before_t=$root/$prefix.fr
export after_t=$root/$prefix.tree.fr
export bpe=${root}/code
export log_file=$root/parse.${prefix}.log
python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}

# for train

# size: 35,762,584
export id=19
export PARSER_PORT=90${id}
export prefix=train.p${id}
export cur=`pwd`
export root=${cur}/../../../raw_data/wmt14_en_fr
export before=$root/$prefix.en
export after=$root/$prefix.tree.en
export before_t=$root/$prefix.fr
export after_t=$root/$prefix.tree.fr
export bpe=${root}/code
export log_file=$root/parse.${prefix}.log
python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}


# convert to BPE
export id=00
export PARSER_PORT=90${id}
export prefix=train.p${id}
export cur=`pwd`
export root=${cur}/../../../raw_data/wmt14_en_fr
export before=$root/$prefix.tree.en.before-bpe
export after=$root/$prefix.tree.bpe.en
export before_t=$root/$prefix.tree.fr
export after_t=$root/$prefix.tree.bpe.fr
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tobpe.log

python -u nstack_process.py --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}
# --------------------------------------------------------------------



# highcpu
# gpu8
export id=08
export id=14
export PARSER_PORT=90${id}
export prefix=train.tok.clean.bpe.32000.p${id}
export cur=`pwd`
export root=${cur}/../../../raw_data/wmt16_ende
export before=$root/$prefix.en
export after=$root/$prefix.tree.en
export bpe=${root}/bpe.32000
export log_file=$root/parse.${prefix}.log
python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${root}/${prefix}.de --after_tgt ${root}/${prefix}.tree.de 2>&1 | tee ${log_file}


# python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${root}/${prefix}.de --after_tgt ${root}/${prefix}.tree.de 2>&1 | tee ${log_file}


# ------ SICK dataset parsing--------------------------------------------------------------

export cur=`pwd`
export root=${cur}/../../../treelstm/data/sick/data
for pref in train test valid; do cp -r $root/$pref.id.txt $root/$pref.id.tree.txt; cp -r $root/$pref.sim.txt $root/$pref.sim.tree.txt; done


export cur=`pwd`
export root=${cur}/../../../treelstm/data/sick/data

for pref in train test valid;
do
for mo in a b;
do
echo $root/$pref.$mo.txt
python -u nstack_process.py --parse_src_only --before $root/$pref.$mo.txt --after $root/$pref.$mo.tree.txt
done
done


export cur=`pwd`
export root=${cur}/../../../treelstm/data/sick/data
export pref=test.id
export pref=test.sym
cp -r $root/$pref.txt $root/$pref.tree.txt

export PARSER_PORT=9000
export prefix=train.a
export prefix=train.b
export prefix=valid.a
export prefix=valid.b
export prefix=test.a
export prefix=test.b

export cur=`pwd`
export root=${cur}/../../../treelstm/data/sick/data
export before=$root/$prefix.txt
export after=$root/$prefix.tree.txt
export log_file=$root/parse.${prefix}.log

python -u nstack_process.py --parse_src_only --before $before --after $after 2>&1 | tee ${log_file}




function pearson(x, y)
  x = x - x:mean()
  y = y - y:mean()
  return x:dot(y) / (x:norm() * y:norm())
end



# ----------------- German ---> English ------------------------

export PARSER_PORT=8001
export prefix=test
export cur=`pwd`
export root=${cur}/../../../raw_data/iwslt14.tokenized.de-en.v2
export before=$root/$prefix.de
export after=$root/$prefix.tree-deen.de
export before_t=$root/$prefix.en
export after_t=$root/$prefix.tree-deen.en
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tree-deen.log

# convert to bpe
export before=$root/$prefix.tree-deen.de.before-bpe
export after=$root/$prefix.tree-deen.de
export before_t=$root/$prefix.en
export after_t=$root/$prefix.tree-deen.en
export log_file=$root/parse.${prefix}.tree-deen.tobpe.log
python -u nstack_process.py --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}

# parse
python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}


# ----------------- French ---> English ------------------------
# raw_data/translate_iwslt_enfr


export PARSER_PORT=8002
export prefix=test.bpe32768
export prefix=valid.bpe32768
export prefix=train.tok.clean.lower.bpe32768

export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt_enfr
export before=$root/$prefix.fr
export after=$root/$prefix.tree-fren.fr
export before_t=$root/$prefix.en
export after_t=$root/$prefix.tree-fren.en
export bpe=${root}/bpe.32768
export log_file=$root/parse.${prefix}.tree-fren.log
# python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}
python -u nstack_process.py  --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}

--ignore_error
source activate pth10
cd /data/nxphi47/projects/nmt/fi_fairseq/src/dptree

cd /export/share/xnguyen/projects/nmt/fi_fairseq/src/dptree
git pull origin master

export idx=02
export PARSER_PORT=70$idx
# export PARSER_PORT=80$idx
export prefix=train.tok.clean.lower.bpe32768.p$idx
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt_enfr
export before=$root/$prefix.fr
export after=$root/$prefix.tree-fren.fr
export before_t=$root/$prefix.en
export after_t=$root/$prefix.tree-fren.en
export bpe=${root}/bpe.32768
export log_file=$root/parse.${prefix}.tree-fren.log
python -u nstack_process.py --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}
python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}

# python -u nstack_process.py  --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}



export PARSER_PORT=70$idx
# export PARSER_PORT=80$idx

train.tok.clean.lower.bpe32768.p01.tree-fren.fr.before-bpe
test.bpe32768.tree.en.before
test.bpe32768.tree.en.before-bpe
test.bpe32768.tree-fren.fr.before-bpe

export prefix=test.bpe32768

export idx=02
export prefix=train.tok.clean.lower.bpe32768.p$idx
# export prefix=valid.bpe32768
# export prefix=test.bpe32768
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt_enfr
export before=$root/$prefix.tree-fren.fr.before-bpe
export after=$root/$prefix.tree-fren.bpe.fr
export before_t=$root/$prefix.tree-fren.en
export after_t=$root/$prefix.tree-fren.bpe.en
export bpe=${root}/bpe.32768
export log_file=$root/parse.${prefix}.tree-fren.bpe.log
python -u nstack_process.py --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}


train.tok.clean.lower.bpe32768.p$idx.tree-fren.bpe
valid.bpe32768.tree-fren.bpe
test.bpe32768.tree-fren.bpe


port=8000 && java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-german.properties -preload tokenize,ssplit,pos,lemma,ner,parse,depp -status_port $port -port $port -timeout 15000000
port=8001 && java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-german.properties -preload tokenize,ssplit,pos,lemma,ner,parse,depp -status_port $port -port $port -timeout 15000000

# ----------------- English - Czech ------------------------
En-Cs

9001

export PARSER_PORT=9001
export prefix=train
export prefix=valid
export prefix=test
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_encs_proc
export before=$root/$prefix.en
export after=$root/$prefix.tree-encs.bpe32k.en
export before_t=$root/$prefix.cs
export after_t=$root/$prefix.tree-encs.bpe32k.cs
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tree-encs.log

# parse
python -u nstack_process.py --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}

# convert to bpe
export before=$root/$prefix.tree-deen.de.before-bpe
export after=$root/$prefix.tree-deen.de
export before_t=$root/$prefix.en
export after_t=$root/$prefix.tree-deen.en
export log_file=$root/parse.${prefix}.tree-deen.tobpe.log
python -u nstack_process.py --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}


# ----------------- Chinese English ------------------------
code  test.en  test.seg.en  test.seg.zh  test.zh  train.en  train.seg.en  train.seg.zh  train.zh
valid.en  valid.seg.en  valid.seg.zh  valid.zh

export PARSER_PORT=6000
export prefix=train
# export prefix=valid
# export prefix=test
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt15_enzh
export before=$root/$prefix.zh
export after=$root/$prefix.tree-zhen.zh
export before_t=$root/$prefix.en
export after_t=$root/$prefix.tree-zhen.en
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tree-encs.log

# parse
python -u nstack_process.py --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}


# ----------------- English - Vietnamese ------------------------
raw_data/translate_envi
bpe.32000       dev-2012-en-vi.tgz  train.bpe32k.vi  train-en-vi      train.vi           tst2012.bpe32k.vi  tst2012.en.bpe32k.en  tst2012.vi.bpe32k.vi  tst2013.bpe32k.vi  tst2013.en.bpe32k.en  tst2013.vi.bpe32k.vi  vocab.bpe.32000.vi
dev-2012-en-vi  train.bpe32k.en     train.en         train-en-vi.tgz  tst2012.bpe32k.en  tst2012.en         tst2012.vi            tst2013.bpe32k.en     tst2013.en         tst2013.vi            vocab.bpe.32000.en

train.bpe32k.tree.en.before-bpe
export PARSER_PORT=9001l
export prefix=train.bpe32k
export prefix=tst2012.bpe32k
export prefix=tst2013.bpe32k
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_envi
export before=$root/$prefix.tree.en.before-bpe
export after=$root/$prefix.tree-afterbpe.en
export before_t=$root/$prefix.tree.vi
export after_t=$root/$prefix.tree-afterbpe.vi
export bpe=${root}/bpe.32000
export log_file=$root/parse.${prefix}.tree-envi.log

# python -u nstack_process.py --ignore_error --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}
# python -u nstack_process.py  --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}

python -u nstack_process.py  --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}





# ----------------- IMDB ---------------------
/data/nxphi47/projects/nmt/raw_data/imdb-dptree-sep
# train.input.raw.proc

export PARSER_PORT=9001
export cur=`pwd`
export root=${cur}/../../../raw_data/imdb-dptree-sep
export pref=dev
export before=$root/$pref.input.raw.proc
export after=$root/$pref.tree.input
export log_file=$root/parse.$pref.log

python -u nstack_process.py  --parse_src_only --before $before --after $after  2>&1 | tee ${log_file}



# ---------- convert to BPE BENEPAR  WMT En-De

export prefix=train.tok.clean.bpe.32000.p0.benepar_en2_large.v2.ende
export prefix=train.tok.clean.bpe.32000.p1.benepar_en2_large.v2.ende
export prefix=train.tok.clean.bpe.32000.p0.benepar_en2_large.v2.ende
# export prefix=newstest2013.tok.bpe.32000.benepar_en2_large.v2.ende
# export prefix=newstest2014.tok.bpe.32000.benepar_en2_large.v2.ende
export src=en
export tgt=de
export cur=`pwd`
export root=${cur}/../../../raw_data/wmt16_ende
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.$tgt
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.w.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.w.$tgt
export bpe=${root}/bpe.32000
export log_file=$root/parse.${prefix}.tree-${src}${tgt}.log
python -u nstack_process.py --workers 16 --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}


# ---------- convert to BPE BENEPAR  WMT En-Fr

# export prefix=newstest2013.tok.bpe.32000.benepar_en2_large.v2.ende
# export prefix=newstest2014.tok.bpe.32000.benepar_en2_large.v2.ende
export prefix=train.tok.clean.bpe.32000.p0.benepar_en2_large.v2.ende
export prefix=train.tok.clean.bpe.32000.p1.benepar_en2_large.v2.ende

export prefix=train.p2.prep250.prep.p00.benepar_en2_large.enfr

# export prefix=train.p2.prep250.prep.p16.benepar_en2_large.enfr
export prefix=valid.benepar_en2_large.enfr
export prefix=test.benepar_en2_large.enfr
export src=en
export tgt=fr
export cur=`pwd`
export root=${cur}/../../../raw_data/wmt14_enfr_v2
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.w.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.w.$tgt
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tree-${src}${tgt}.log
python -u nstack_process.py --workers 4 --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}

export src=en
export tgt=fr
export cur=`pwd`
export root=${cur}/../../../raw_data/wmt14_enfr_v2
export bpe=${root}/code
# for idx in 00 01 02 03
# for idx in 04 05 06 07
# for idx in 08 09 10 11
# for idx in 12 13 14 15 16
for idx in 11 12 13 14 15
do
# export prefix=train.p1.prep250.prep.p${idx}.benepar_en2_large.enfr
export prefix=train.p2.prep250.prep.p${idx}.benepar_en2_large.enfr
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.w.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.w.$tgt
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tree-${src}${tgt}.log
python -u nstack_process.py --workers 8 --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}
done


# ---------- convert to IWSLT BPE BENEPAR En-De
export src=en
export tgt=de
export prefix=train.benepar_en2_large.ende
# export prefix=test.benepar_en2_large.ende
# export prefix=valid.benepar_en2_large.ende
export cur=`pwd`
export root=${cur}/../../../raw_data/iwslt14.tokenized.de-en.v2
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.$tgt
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tree-${src}${tgt}.log
python -u nstack_process.py  --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}


# ---------- convert to IWSLT BPE BENEPAR Deen
export src=de
export tgt=en
export prefix=train.benepar_de.deen
# export prefix=valid.benepar_de.deen
# export prefix=test.benepar_de.deen
export cur=`pwd`
export root=${cur}/../../../raw_data/iwslt14.tokenized.de-en.v2
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.$tgt
export bpe=${root}/code
export log_file=$root/parse.${prefix}.tree-${src}${tgt}.log
python -u nstack_process.py  --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}



# ---------- convert to IWSLT BPE BENEPAR En-Fr
export src=en
export tgt=fr
export prefix=train.tok.clean.lower.bpe32768.benepar_en2_large.enfr
export prefix=test.bpe32768.benepar_en2_large.enfr
export prefix=valid.bpe32768.benepar_en2_large.enfr
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt_enfr
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.$tgt
export bpe=${root}/bpe.32768
export log_file=$root/parse.${prefix}.tree-${src}${tgt}.log
python -u nstack_process.py  --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}



# ---------- convert to IWSLT BPE BENEPAR Fr-En
export src=fr
export tgt=en
export prefix=train.tok.clean.lower.bpe32768.benepar_fr.fren
# export prefix=valid.bpe32768.benepar_fr.fren
# export prefix=test.bpe32768.benepar_fr.fren
export cur=`pwd`
export root=${cur}/../../../raw_data/translate_iwslt_enfr
export before=$root/$prefix.$src
export after=$root/$prefix.bpe.$src
export before_t=$root/$prefix.$tgt
export after_t=$root/$prefix.bpe.$tgt
export bpe=${root}/bpe.32768
export log_file=$root/parse.${prefix}.tree-${src}${tgt}.log
python -u nstack_process.py  --convert_bpe --bpe_code ${bpe} --bpe_tree --before $before --after $after --before_tgt ${before_t} --after_tgt ${after_t} 2>&1 | tee ${log_file}




"""



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before_src')
    parser.add_argument('--after_src')
    parser.add_argument('--before_tgt')
    parser.add_argument('--after_tgt')
    parser.add_argument('--bpe_code')
    parser.add_argument('--raw_vocab', default = None)
    parser.add_argument('--ignore_error', action = 'store_true')
    parser.add_argument('--workers', default = 0, type = int)
    parser.add_argument('--parser_host', default = 'localhost', type = str)
    parser.add_argument('--parser_port', default = 9000, type = int)

    args = parser.parse_args()

    builder = SeqTreeBuilder(
        transform = True,
        bpe_code = args.bpe_code,
        ignore_error = args.ignore_error,
        parser_host = args.parser_host,
        parser_port = args.parser_port,
    )
    builder.export_text_to_tree_strings(
        input_file = args.before_src,
        output_file = args.after_src,
        num_workers = args.workers,
        tgt_file = args.before_tgt,
        tgt_out_file = args.after_tgt,
    )
