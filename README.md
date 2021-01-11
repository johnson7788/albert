ALBERT
======

***************New March 28, 2020 ***************

Add a colab [tutorial](https://github.com/google-research/albert/blob/master/albert_glue_fine_tuning_tutorial.ipynb) to run fine-tuning for GLUE datasets.

***************New January 7, 2020 ***************

v2 TF-Hub模型现在应该可以在TF 1.15上使用，因为我们从图中删除了局部Einsum op。 请参阅下面的更新的TF-Hub链接。


***************New December 30, 2019 ***************

中文模型发布。 感谢[CLUE team ](https://github.com/CLUEbenchmark/CLUE)提供的训练数据。是v1版本的。

- [Base](https://storage.googleapis.com/albert_models/albert_base_zh.tar.gz)
- [Large](https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz)
- [Xlarge](https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz)
- [Xxlarge](https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz)

ALBERT模型的版本2已发布。 

- Base: [[Tar file](https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_base/3)]
- Large: [[Tar file](https://storage.googleapis.com/albert_models/albert_large_v2.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_large/3)]
- Xlarge: [[Tar file](https://storage.googleapis.com/albert_models/albert_xlarge_v2.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_xlarge/3)]
- Xxlarge: [[Tar file](https://storage.googleapis.com/albert_models/albert_xxlarge_v2.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_xxlarge/3)]

在此版本中，我们对所有模型应用“无dropout”，“其他训练数据”和“长时间训练”策略。 我们训练基于ALBERT的10M步，训练其他模型3M的步。

与v1模型的结果比较如下： 


|                | Average  | SQuAD1.1 | SQuAD2.0 | MNLI     | SST-2    | RACE     |
|----------------|----------|----------|----------|----------|----------|----------|
|V2              |
|ALBERT-base     |82.3      |90.2/83.2 |82.1/79.3 |84.6      |92.9      |66.8      |
|ALBERT-large    |85.7      |91.8/85.2 |84.9/81.8 |86.5      |94.9      |75.2      |
|ALBERT-xlarge   |87.9      |92.9/86.4 |87.9/84.1 |87.9      |95.4      |80.7      |
|ALBERT-xxlarge  |90.9      |94.6/89.1 |89.8/86.9 |90.6      |96.8      |86.8      |
|V1              |
|ALBERT-base     |80.1      |89.3/82.3 | 80.0/77.1|81.6      |90.3      | 64.0     |
|ALBERT-large    |82.4      |90.6/83.9 | 82.3/79.4|83.5      |91.7      | 68.5     |
|ALBERT-xlarge   |85.5      |92.5/86.1 | 86.1/83.1|86.4      |92.4      | 74.8     |
|ALBERT-xxlarge  |91.0      |94.8/89.3 | 90.2/87.4|90.8      |96.9      | 86.5     |

比较表明，对于基于ALBERT的，ALBERT-large的和ALBERT-xlarge的，v2优于v1，表明应用以上三种策略的重要性。 
平均而言，由于以下两个原因，ALBERT-xxlarge比v1稍差：
1)训练额外的1.5M步(这两个模型之间的唯一区别是训练1.5M步和3M步)并没有带来明显的性能改进。 
2)对于v1，我们在BERT，Roberta和XLnet给定的参数集中进行了一些超参数搜索。 
对于v2，我们仅采用v1中的参数，除了RACE，我们使用的学习率是1e-5和0 [ALBERT DR](https://arxiv.org/pdf/1909.11942.pdf) (ALBERT的dropout rate 微调)。
原始(v1)RACE超参数将导致v2模型的模型分歧。 鉴于下游任务对微调超参数敏感，因此我们应谨慎对待所谓的细微改进。 

ALBERT是BERT的“精简版”版本，它是一种流行的无监督语言表示学习算法。 
ALBERT使用参数缩减技术，可进行大规模配置，克服先前的内存限制并在模型降级方面实现更好的性能。 


有关该算法的技术说明，请参见我们的论文： 

[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut

Release Notes
=============

- Initial release: 10/9/2019

Results
=======

ALBERT在GLUE基准测试结果上使用单模型设置的性能 
dev:

| Models            | MNLI     | QNLI     | QQP      | RTE      | SST      | MRPC     | CoLA     | STS      |
|-------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| BERT-large        | 86.6     | 92.3     | 91.3     | 70.4     | 93.2     | 88.0     | 60.6     | 90.0     |
| XLNet-large       | 89.8     | 93.9     | 91.8     | 83.8     | 95.6     | 89.2     | 63.6     | 91.8     |
| RoBERTa-large     | 90.2     | 94.7     | **92.2** | 86.6     | 96.4     | **90.9** | 68.0     | 92.4     |
| ALBERT (1M)       | 90.4     | 95.2     | 92.0     | 88.1     | 96.8     | 90.2     | 68.7     | 92.7     |
| ALBERT (1.5M)     | **90.8** | **95.3** | **92.2** | **89.2** | **96.9** | **90.9** | **71.4** | **93.0** |

使用单模型的ALBERT-xxl在SQuaD和RACE基准测试中的性能 
setup:

|Models                    | SQuAD1.1 dev  | SQuAD2.0 dev  | SQuAD2.0 test | RACE test (Middle/High) |
|--------------------------|---------------|---------------|---------------|-------------------------|
|BERT-large                | 90.9/84.1     | 81.8/79.0     | 89.1/86.3     | 72.0 (76.6/70.1)        |
|XLNet                     | 94.5/89.0     | 88.8/86.1     | 89.1/86.3     | 81.8 (85.5/80.2)        |
|RoBERTa                   | 94.6/88.9     | 89.4/86.5     | 89.8/86.8     | 83.2 (86.5/81.3)        |
|UPM                       | -             | -             | 89.9/87.2     | -                       |
|XLNet + SG-Net Verifier++ | -             | -             | 90.1/87.2     | -                       |
|ALBERT (1M)               | 94.8/89.2     | 89.9/87.2     | -             | 86.0 (88.2/85.1)        |
|ALBERT (1.5M)             | **94.8/89.3** | **90.2/87.4** | **90.9/88.1** | **86.5 (89.0/85.5)**    |


Pre-trained Models
==================
TF-Hub modules are available:

- Base: [[Tar file](https://storage.googleapis.com/albert_models/albert_base_v1.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_base/1)]
- Large: [[Tar file](https://storage.googleapis.com/albert_models/albert_large_v1.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_large/1)]
- Xlarge: [[Tar file](https://storage.googleapis.com/albert_models/albert_xlarge_v1.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_xlarge/1)]
- Xxlarge: [[Tar file](https://storage.googleapis.com/albert_models/albert_xxlarge_v1.tar.gz)] [[TF-Hub](https://tfhub.dev/google/albert_xxlarge/1)]

Example usage of the TF-Hub module in code:

```
tags = set()
if is_training:
  tags.add("train")
albert_module = hub.Module("https://tfhub.dev/google/albert_base/1", tags=tags,
                           trainable=True)
albert_inputs = dict(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids)
albert_outputs = albert_module(
    inputs=albert_inputs,
    signature="tokens",
    as_dict=True)

# If you want to use the token-level output, use
# albert_outputs["sequence_output"] instead.
output_layer = albert_outputs["pooled_output"]
```

该repository中的大多数微调脚本都通过--albert_hub_module_handle标志来支持TF-hub模块。 

预训练设置
=========================
要预训练ALBERT，请使用`run_pretraining.py`： 

```
pip install -r albert/requirements.txt
python -m albert.run_pretraining \
    --input_file=... \
    --output_dir=... \
    --init_checkpoint=... \
    --albert_config_file=... \
    --do_train \
    --do_eval \
    --train_batch_size=4096 \
    --eval_batch_size=64 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=125000 \
    --num_warmup_steps=3125 \
    --save_checkpoints_steps=5000
```

Fine-tuning on GLUE
===================
要在GLUE上微调和评估经过预训练的ALBERT，请参见便捷脚本`run_glue.sh`。

Lower-level用例可能要直接使用`run_classifier.py`脚本。
`run_classifier.py`脚本用于微调和评估单个GLUE基准测试任务(例如MNLI)上的ALBERT： 

```
pip install -r albert/requirements.txt
python -m albert.run_classifier \
  --data_dir=... \
  --output_dir=... \
  --init_checkpoint=... \
  --albert_config_file=... \
  --spm_model_file=... \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --max_seq_length=128 \
  --optimizer=adamw \
  --task_name=MNLI \
  --warmup_step=1000 \
  --learning_rate=3e-5 \
  --train_step=10000 \
  --save_checkpoints_steps=100 \
  --train_batch_size=128
```

可以在`run_glue.sh` 中找到每个GLUE任务的默认标志值。 

您可以通过设置例如`--albert_hub_module_handle=https://tfhub.dev/google/albert_base/1` 使用TF-Hub模块, 而是原始checkpoint开始微调模型代替 `--init_checkpoint`.

您可以在tar文件或tf-hub模块的asset文件夹下找到spm_model_file。 模型文件的名称为“30k-clean.model”。

评估后，脚本应报告如下输出： 

```
***** Eval results *****
  global_step = ...
  loss = ...
  masked_lm_accuracy = ...
  masked_lm_loss = ...
  sentence_order_accuracy = ...
  sentence_order_loss = ...
```

Fine-tuning on SQuAD
====================
要在SQuAD v1上微调和评估预训练模型，请使用 
`run_squad_v1.py` script:

```
pip install -r albert/requirements.txt
python -m albert.run_squad_v1 \
  --albert_config_file=... \
  --output_dir=... \
  --train_file=... \
  --predict_file=... \
  --train_feature_file=... \
  --predict_feature_file=... \
  --predict_feature_left_file=... \
  --init_checkpoint=... \
  --spm_model_file=... \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train=true \
  --do_predict=true \
  --train_batch_size=48 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=5000 \
  --n_best_size=20 \
  --max_answer_length=30
```

您可以通过设置例如
`--albert_hub_module_handle=https://tfhub.dev/google/albert_base/1` 
 从TF-Hub模块而不是`--init_checkpoint`原始checkpoint开始微调模型 

For SQuAD v2, use the `run_squad_v2.py` script:

```
pip install -r albert/requirements.txt
python -m albert.run_squad_v2 \
  --albert_config_file=... \
  --output_dir=... \
  --train_file=... \
  --predict_file=... \
  --train_feature_file=... \
  --predict_feature_file=... \
  --predict_feature_left_file=... \
  --init_checkpoint=... \
  --spm_model_file=... \
  --do_lower_case \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64 \
  --do_train \
  --do_predict \
  --train_batch_size=48 \
  --predict_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --warmup_proportion=.1 \
  --save_checkpoints_steps=5000 \
  --n_best_size=20 \
  --max_answer_length=30
```

You can fine-tune the model starting from TF-Hub modules instead of raw checkpoints by setting e.g.
`--albert_hub_module_handle=https://tfhub.dev/google/albert_base/1` instead
of `--init_checkpoint`.

Fine-tuning on RACE
===================
For RACE, use the `run_race.py` script:

```
pip install -r albert/requirements.txt
python -m albert.run_race \
  --albert_config_file=... \
  --output_dir=... \
  --train_file=... \
  --eval_file=... \
  --data_dir=...\
  --init_checkpoint=... \
  --spm_model_file=... \
  --max_seq_length=512 \
  --max_qa_length=128 \
  --do_train \
  --do_eval \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --learning_rate=1e-5 \
  --train_step=12000 \
  --warmup_step=1000 \
  --save_checkpoints_steps=100
```

You can fine-tune the model starting from TF-Hub modules instead of raw
checkpoints by setting e.g.
`--albert_hub_module_handle=https://tfhub.dev/google/albert_base/1` instead
of `--init_checkpoint`.

SentencePiece
=============
生成句子单词表的命令： 

```
spm_train \
--input all.txt --model_prefix=30k-clean --vocab_size=30000 --logtostderr
--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1
--control_symbols=[CLS],[SEP],[MASK]
--user_defined_symbols="(,),\",-,.,–,£,€"
--shuffle_input_sentence=true --input_sentence_size=10000000
--character_coverage=0.99995 --model_type=unigram
```
