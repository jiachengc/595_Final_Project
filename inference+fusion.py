import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator

from tqdm.autonotebook import tqdm
import collections
from collections import Counter

import gc

import warnings
warnings.filterwarnings("ignore")

max_answer_length = 30

test = pd.read_csv('../input/chaii-hindi-and-tamil-question-answering/test.csv')
test_dataset = Dataset.from_pandas(test)

#test.head()
#%env WANDB_DISABLED=True

data_collator = default_data_collator


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        predictions[example["id"]] = best_answer["text"]

    return predictions

def prepare_validation_features(examples, tokenizer=None, pad_on_right=True):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def run_blend(tok_checkpoint, model_checkpoints, batch_size, max_length, doc_stride, pp=True):
    args = TrainingArguments(
        'output_dir',
        per_device_eval_batch_size=batch_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_checkpoint)
    pad_on_right = tokenizer.padding_side == "right"
    test_features = test_dataset.map(
        prepare_validation_features,
        batched=True,
        remove_columns=test_dataset.column_names,
        fn_kwargs={'tokenizer': tokenizer, 'pad_on_right': pad_on_right}
    )
    test_feats_small = test_features.map(lambda example: example, remove_columns=['example_id', 'offset_mapping'])

    model = AutoModelForQuestionAnswering.from_pretrained(tok_checkpoint)

    start_logits = []
    end_logits = []

    for model_checkpoint in model_checkpoints:
        model.load_state_dict(torch.load(model_checkpoint))
        model.eval()
        trainer = Trainer(
            model,
            args,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        test_predictions = trainer.predict(test_feats_small)
        start_logits.append(test_predictions.predictions[0])
        end_logits.append(test_predictions.predictions[1])

    start_logits = sum(start_logits) / len(model_checkpoints)
    end_logits = sum(end_logits) / len(model_checkpoints)
    predictions = (start_logits, end_logits)

    test_features.set_format(type=test_features.format["type"], columns=list(test_features.features.keys()))
    final_test_predictions = postprocess_qa_predictions(test_dataset, test_features, predictions, tokenizer)

    return final_test_predictions

def postuning(t):
    punctuation = '!"#&\'()*+,-./:;=?@[\\]^_`{|}~'
    tn = t.strip(punctuation)
    # " "
    if '"' in tn and tn + '"' in t: tn = tn + '"'
    if '"' in tn and '"' + tn in t: tn = '"' + tn
    # ' '
    if "'" in tn and tn + "'" in t: tn = tn + "'"
    if "'" in tn and "'" + tn in t: tn = "'" + tn
    # ( )
    if '(' in tn and tn + ')' in t: tn = tn + ')'
    if ')' in tn and '(' + tn in t: tn = '(' + tn
    # - -
    if '-' in tn and tn + '-' in t: tn = tn + '-'
    if '-' in tn and '-' + tn in t: tn = '-' + tn
    # ई.
    if tn[-1] == 'ई' and 'ई.' in t: tn = tn + '.'
    # ई.पू.
    if tn[-4:] == 'ई.पू' and 'ई.पू.' in t: tn = tn + '.'
    # सें
    if tn[-3:] == 'सें' and 'सें.' in t: tn = tn + '.'
    # वर्ग कि.मी
    if tn[-10:] == 'वर्ग कि.मी' and 'वर्ग कि.मी.' in t: tn = tn + '.'
    # सें
    if tn[-10:] == 'एस. पी. जी' and 'एस. पी. जी.' in t: tn = tn + '.'
    # सें
    if tn[-5:] == 'கி.மீ' and 'கி.மீ.' in t: tn = tn + '.'
    return tn

blendxlm = [
    '../input/XLM_Roberta_large_1/pytorch_model_0.bin',
    '../input/XLM_Roberta_large_2/pytorch_model_0.bin'
]
blendmur = [
    '../input/MuRIL_large_1/pytorch_model_0.bin',
]
blendrem = [
    '../input/RemBert_1/pytorch_model_0.bin',
    '../input/RemBert_2/pytorch_model_0.bin',
]

models = [
    ('xlmr1',
    '../input/xlm-roberta-squad2/deepset/xlm-roberta-large-squad2',
    blendxlm,
    32,
    384,
    128,
    True),
    ('muri1',
    '../input/muril-large-pt/muril-large-cased',
    blendmur,
    32,
    384,
    128,
    False),
    ('remb1',
    '../input/rembert-pt',
    blendrem,
    32,
    384,
    128,
    True),
]

sub = pd.read_csv('../input/chaii-hindi-and-tamil-question-answering/sample_submission.csv')

for (name, tok_checkpoint, model_checkpoints, batch_size, max_length, doc_stride, pp) in models:
    preds = run_blend(tok_checkpoint, model_checkpoints, batch_size, max_length, doc_stride)
    sub[name] = sub['id'].apply(lambda r: preds[r])
    if pp:
        sub[name] = sub[name].apply(postuning)

def count(r):
    words = []
    for col in r:
        words.extend(col.split(' '))
    return Counter(words)

sub['word_dict'] = sub[['xlmr1', 'muri1', 'remb1']].apply(count, axis=1)
sub['PredictionString'] = sub['muri1']
sub['overlap'] = sub['word_dict'].apply(lambda x: ' '.join([w for w in x.keys() if x[w] > 1]))
sub['PredictionString'][sub['overlap'] != ''] = sub['overlap'][sub['overlap'] != '']
sub = sub[['id', 'PredictionString']]
sub.to_csv('submission.csv', index=False)









