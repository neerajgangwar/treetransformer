An implementation of "Tree-Structured Attention with Hierarchical Accumulation" ([OpenReview](https://openreview.net/forum?id=HJxK5pEYvr)). For the author's implementation, see https://github.com/nxphi47/tree_transformer. Also, consider citing their work!

The repository borrows the model from the author's implementation and migrates the code to use fairseq v0.12.3.

Environment Setup
-----------------
`conda` or `mamba` may be used to setup the environment as follows
```bash
conda env create -f environment.yml -n treetx
```

Preparing the Datasets
----------------------
### 1. Download the Dataset
The following command can be used to download IWSLT'14 En-De dataset.
```bash
bash scripts/prepare-iwslt14.sh
```
This script does a bunch of post-processing on the datasets and will save the final output to `iwslt14.tokenized.de-en` directory. The directory will have dataset files and BPE codes as shown below
```bash
$ ls iwslt14.tokenized.de-en
code  test.de  test.en  tmp  train.de  train.en  valid.de  valid.en
```

### 2. Constituency Parse Trees
The next step is to get constituency parse trees for the input sequences. First, download and run [Stanford CoreNLP Parser](https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK). Validate for the correct language (we need to parse the source text into constituency parse trees). Then, use the following script to get constituency parse tree for train, valid, and test files.
```bash
BEFORE_SRC=iwslt14.tokenized.de-en/train.en
AFTER_SRC=iwslt14.tokenized.de-en/train.en-de.tree.en
BEFORE_TGT=iwslt14.tokenized.de-en/train.de
AFTER_TGT=iwslt14.tokenized.de-en/train.en-de.tree.de
BPE_CODE=iwslt14.tokenized.de-en/code
CORENLP_PARSER_HOST=127.0.0.1
CORENLP_PARSER_PORT=9000
python scripts/parse_nmt.py \
    --before_src $BEFORE_SRC \
    --after_src $AFTER_SRC \
    --before_tgt $BEFORE_TGT \
    --after_tgt $AFTER_TGT \
    --bpe_code $BPE_CODE \
    --ignore_error \
    --workers 32 \
    --parser_host $CORENLP_PARSER_HOST \
    --parser_port $CORENLP_PARSER_PORT
```
This script needs to be run for `train`, `valid`, and `test`, resulting in the following files:
- `train.en-de.tree.en` and `train.en-de.tree.de`
- `valid.en-de.tree.en` and `valid.en-de.tree.de`
- `test.en-de.tree.en` and `test.en-de.tree.de`


Running the Model
-----------------
Like other fairseq models, this runs in three stages - preprocessing, training, and evaluation.

### Preprocessing
To preprocess the dataset, run the following script
```bash
python scripts/preprocess.py \
    --user-dir ./src \
    --source-lang en \
    --target-lang de \
    --trainpref train.en-de.tree \
    --validpref valid.en-de.tree \
    --testpref test.en-de.tree \
    --destdir datasets-bin/iwslt.en-de.tree \
    --workers 20 \
    --joined-dictionary
```
The tokenized dataset and dictionary files will be saved inside `datasets-bin/iwslt.en-de.tree`.

### Training
Run the following command to start the training.
```bash
SEED=42
OUTDIR=trained_models/iwslt-en-de-tree
fairseq-train datasets-bin/iwslt.en-de.tree \
    --user-dir ./src \
    --arch dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier \
    --task tree_translation \
    --source-lang en \
    --target-lang de \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 5e-4 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --dataset-impl lazy \
    --left-pad-source \
    --fp16 \
    --max-update 61000 \
    --keep-last-epochs 20  \
    --seed $SEED \
    --save-dir $OUTDIR \
    --log-format json \
    --log-file $OUTDIR/training.log
```

### Evaluation
Use the following command to evaluate the trained model.
```bash
fairseq-generate datasets-bin/iwslt.en-de.tree \
    --user-dir ./src \
    --task tree_translation \
    --path $OUTDIR/checkpoint_best.pt \
    --max-tokens 4096 \
    --beam 5 \
    --remove-bpe \
    --scoring sacrebleu \
    --left-pad-source \
    --tokenizer moses \
    --source-lang en \
    --target-lang de \
    --dataset-impl lazy \
    --results-path $OUTDIR \
    --fp16
```

### Averaging Checkpoints
If you need to average multiple saved checkpoints, run `scripts/average_checkpoints.py`. To evaluate the resulting checkpoint, run evaluation with `--path` as the resulting checkpoint.
