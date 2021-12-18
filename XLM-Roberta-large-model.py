
import random
import torch
torch.cuda.is_available()
import torch
from transformers import AutoTokenizer
import pandas as pd
from transformers import default_data_collator
from sklearn import model_selection
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from tqdm.autonotebook import tqdm
import warnings
warnings.filterwarnings("ignore")
from transformers import RobertaPreTrainedModel, RobertaModel, EvalPrediction
from collections import defaultdict
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Any, Dict, List, NewType, Optional, Tuple
from transformers import Trainer, is_torch_tpu_available
import numpy as np
from transformers.trainer_utils import PredictionOutput
from tqdm.auto import tqdm
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.file_utils import is_datasets_available
from string import punctuation
import collections
import json
import logging
import os

random.seed(100)
folder_name = 'XLM_Roberta_large'
total_folder_name = folder_name
SAMPLE = False
logger = logging.getLogger(__name__)

model_checkpoint = 'xlm-roberta-large'
batch_size = 8
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_length = 384
doc_stride = 192
pad_on_right = tokenizer.padding_side == "right"
n_folds = 3

version_2_with_negative = False
n_best_size = 20
max_answer_length = 30
question_column_name = "question"
context_column_name = "context"
answer_column_name = "answers"

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class SubTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, predict_examples, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                predict_dataloader,
                description="Prediction",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(predict_examples, predict_dataset, output.predictions, "predict")
        metrics = self.compute_metrics(predictions)

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=metrics)

class MainTrainer(SubTrainer):

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        train_sampler = SequentialSampler(self.train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def prediction_processing(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
):
    if len(predictions) != 2:
        raise ValueError("`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(f"Got {len(predictions[0])} predictions and {len(features)} features.")

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    logger.setLevel(log_level)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            raise EnvironmentError(f"{output_dir} is not a directory.")

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def processing_results(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = prediction_processing(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=version_2_with_negative,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
    )
    if version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

metric = load_metric("squad_v2" if version_2_with_negative else "squad")

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)

def prepare_train_features(examples, max_length=384, doc_stride=192):
    examples["question"] = [q.lstrip() for q in examples["question"]]

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

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples, max_length=384, doc_stride=128):
    examples["question"] = [q.lstrip() for q in examples["question"]]

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

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=2021)
    for fold_num, (t_, v_) in enumerate(kf.split(X=data, y=data.language.values)):
        data.loc[v_, "kfold"] = fold_num
    return data


def convert_answers(row):
    return {"answer_start": [row[0]], "text": [row[1]]}

def negative_sampling(examples, ratio=0.1):
    def _sample(pos):
        if pos != 0:
            return True
        else:
            return random.random() < ratio

    indices = [i for i, x in enumerate(examples['start_positions']) if _sample(x)]

    for key in examples.keys():
        examples[key] = [x for i, x in enumerate(examples[key]) if i in indices]

    return examples

def jaccard(row):
    str1 = row[0]
    str2 = row[1]
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def postuning(s):
    s = " ".join(s.split())
    s = s.strip(punctuation)
    return s

def prepare_datasets(fold, seed, sample=False):
    train = pd.read_csv("data/train.csv")
    threeds = pd.read_csv('data/chaii-mlqa-xquad-5folds.csv')
    threeds["answers"] = threeds[["answer_start", "answer_text"]].apply(convert_answers, axis=1)

    hindi = threeds[threeds.src != 'chaii'].reset_index(drop=True)
    chaii = threeds[threeds.src == 'chaii'].reset_index(drop=True)
    chaii = pd.merge(chaii, train[['id', 'context', 'question']], how='left', on=['context', 'question'])
    del chaii['fold']

    if sample == True: chaii = chaii.sample(n=20, random_state=42).reset_index(drop=True)

    chaii = create_folds(chaii, 3)

    chaii_train = chaii[chaii.kfold != fold]
    chaii_valid = chaii[chaii.kfold == fold]

    chaii_train_ds = Dataset.from_pandas(chaii_train)
    chaii_valid_ds = Dataset.from_pandas(chaii_valid)

    tokenized_chaii_train_strid1 = chaii_train_ds.map(prepare_train_features,
                                                      fn_kwargs={'max_length': 256, 'doc_stride': 112}, batched=True,
                                                      remove_columns=chaii_train_ds.column_names, batch_size=32)
    tokenized_chaii_train_strid2 = chaii_train_ds.map(prepare_train_features,
                                                      fn_kwargs={'max_length': 384, 'doc_stride': 192}, batched=True,
                                                      remove_columns=chaii_train_ds.column_names, batch_size=32)
    tokenized_chaii_train_strid3 = chaii_train_ds.map(prepare_train_features,
                                                      fn_kwargs={'max_length': 384, 'doc_stride': 128}, batched=True,
                                                      remove_columns=chaii_train_ds.column_names, batch_size=32)
    validation_features = chaii_valid_ds.map(prepare_validation_features,
                                             fn_kwargs={'max_length': 384, 'doc_stride': 128}, batched=True,
                                             remove_columns=chaii_valid_ds.column_names, batch_size=32)

    if sample == True:
        trn = concatenate_datasets([tokenized_chaii_train_strid1, tokenized_chaii_train_strid2])
        return (trn, validation_features, chaii_valid_ds)

    hindi_ds = Dataset.from_pandas(hindi)
    tokenized_hindi_strid1 = hindi_ds.map(prepare_train_features, fn_kwargs={'max_length': 256, 'doc_stride': 112},
                                          batched=True, remove_columns=hindi_ds.column_names, batch_size=32)
    tokenized_hindi_strid2 = hindi_ds.map(prepare_train_features, fn_kwargs={'max_length': 384, 'doc_stride': 128},
                                          batched=True, remove_columns=hindi_ds.column_names, batch_size=32)

    squad = load_dataset("squad")
    tokenized_squad = squad['train'].map(prepare_train_features, fn_kwargs={'max_length': 256, 'doc_stride': 128},
                                         batched=True, remove_columns=squad['train'].column_names, batch_size=32)

    tydi = pd.read_csv('data/tydiqa_train.csv')
    tydi_bete_all = tydi[
        (tydi.language == 'bengali') | (tydi.language == 'telugu') | (tydi.language == 'english')].reset_index(
        drop=True)

    tydi_bete_all["answers"] = tydi_bete_all[["answer_start", "answer_text"]].apply(convert_answers, axis=1)
    tydi_bete_all_ds = Dataset.from_pandas(tydi_bete_all)
    tokenized_tydi_bete_all_strid1 = tydi_bete_all_ds.map(prepare_train_features,
                                                          fn_kwargs={'max_length': 256, 'doc_stride': 128},
                                                          batched=True, remove_columns=tydi_bete_all_ds.column_names,
                                                          batch_size=32)
    tokenized_tydi_bete_all_strid2 = tydi_bete_all_ds.map(prepare_train_features,
                                                          fn_kwargs={'max_length': 384, 'doc_stride': 192},
                                                          batched=True, remove_columns=tydi_bete_all_ds.column_names,
                                                          batch_size=32)

    nq = pd.read_csv('data/nq_small.csv')
    nq = nq.rename(columns={'answer': 'answer_text'})
    nq["answers"] = nq[["answer_start", "answer_text"]].apply(convert_answers, axis=1)
    nq_ds = Dataset.from_pandas(nq)
    tokenized_nq_strid = nq_ds.map(prepare_train_features, fn_kwargs={'max_length': 384, 'doc_stride': 192},
                                   batched=True, remove_columns=nq_ds.column_names, batch_size=32)

    tokenized_tydi_bete_all_strid1 = tokenized_tydi_bete_all_strid1.map(negative_sampling, batched=True, batch_size=32)
    tokenized_tydi_bete_all_strid2 = tokenized_tydi_bete_all_strid2.map(negative_sampling, batched=True, batch_size=32)
    tokenized_nq_strid = tokenized_nq_strid.map(negative_sampling, fn_kwargs={'ratio': 0.06}, batched=True,
                                                batch_size=32)
    tokenized_chaii_train_strid1 = tokenized_chaii_train_strid1.map(negative_sampling, fn_kwargs={'ratio': 0.1},
                                                                    batched=True, batch_size=32)
    tokenized_chaii_train_strid2 = tokenized_chaii_train_strid2.map(negative_sampling, fn_kwargs={'ratio': 0.2},
                                                                    batched=True, batch_size=32)
    tokenized_chaii_train_strid3 = tokenized_chaii_train_strid3.map(negative_sampling, fn_kwargs={'ratio': 0.3},
                                                                    batched=True, batch_size=32)

    ep1 = concatenate_datasets([tokenized_tydi_bete_all_strid1, tokenized_squad, tokenized_hindi_strid1,
                                tokenized_chaii_train_strid1]).shuffle(seed=seed)
    ep2 = concatenate_datasets([tokenized_tydi_bete_all_strid2, tokenized_nq_strid, tokenized_hindi_strid2,
                                tokenized_chaii_train_strid2]).shuffle(seed=seed)
    tokenized_train_all = concatenate_datasets([ep1, ep2, tokenized_chaii_train_strid3])

    return (tokenized_train_all, validation_features, chaii_valid_ds)

InputDataClass = NewType("InputDataClass", Any)

def random_mask_data_collator(features: List[InputDataClass], mlm_probability=0.1) -> Dict[str, Any]:
    batch = tokenizer.pad(features, return_tensors="pt")

    probability_matrix = torch.full(batch['input_ids'].shape, mlm_probability)
    special_tokens_mask = [[
        1 if x in [0, 1, 2] else 0 for x in row.tolist()
    ] for row in batch['input_ids']]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    batch['input_ids'][masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return batch

data_collator = random_mask_data_collator

for fold in range(0, 3):
    print(f'Training fold {fold}')
    train_ds, valid_ds, valid_examples = prepare_datasets(fold, 100, sample=SAMPLE)
    print((len(train_ds), len(valid_ds), len(valid_examples)))
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    args = TrainingArguments(
        f"chaii-qa-{folder_name}-fold{fold}",
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=1000,
        save_steps=1000,
        save_strategy="steps",
        learning_rate=1e-5,
        gradient_accumulation_steps=4,
        warmup_ratio=0.2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to='none',
        save_total_limit=4
    )
    trainer = MainTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        eval_examples=valid_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=processing_results,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(f"{total_folder_name}/final_fold{fold}")
    fineval = trainer.evaluate()
    print(f'final eval fold {fold}:')
    print(fineval)
    raw_predictions = trainer.predict(valid_ds, valid_examples)
    final_predictions = defaultdict()
    for x in raw_predictions.predictions: final_predictions[x['id']] = x['prediction_text']
    references = [
        {"id": ex["id"], "context": ex["context"], "question": ex["question"], "answer": ex["answers"]['text'][0]} for
        ex in valid_examples]

    res = pd.DataFrame(references)
    res['prediction'] = res['id'].apply(lambda r: final_predictions[r])

    res['jaccard'] = res[['answer', 'prediction']].apply(jaccard, axis=1)
    res['postuned'] = res['prediction'].apply(postuning)
    res['pjaccard'] = res[['answer', 'postuned']].apply(jaccard, axis=1)

    print(f'Fold: {fold} Jaccard normal: {res.jaccard.mean()} jaccard postuned: {res.pjaccard.mean()}')

data_collator = default_data_collator

model_checkpoint = 'XLM_Roberta_large/final_fold1'
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
trainer = MainTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=processing_results,
    compute_metrics=compute_metrics,
)
raw_predictions = trainer.predict(valid_ds, valid_examples)
final_predictions = defaultdict()
for x in raw_predictions.predictions: final_predictions[x['id']] = x['prediction_text']
res = pd.DataFrame(references)
res['prediction'] = res['id'].apply(lambda r: final_predictions[r])
res['jaccard'] = res[['answer', 'prediction']].apply(jaccard, axis=1)
res['postuned'] = res['prediction'].apply(postuning)
res['pjaccard'] = res[['answer', 'postuned']].apply(jaccard, axis=1)
res.jaccard.mean(), res.pjaccard.mean()

model_checkpoint = 'chaii-qa-XLM_Roberta_large-fold2/checkpoint-8000'
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
trainer = MainTrainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=processing_results,
    compute_metrics=compute_metrics,
)
raw_predictions = trainer.predict(valid_ds, valid_examples)
final_predictions = defaultdict()
for x in raw_predictions.predictions: final_predictions[x['id']] = x['prediction_text']
res = pd.DataFrame(references)
res['prediction'] = res['id'].apply(lambda r: final_predictions[r])
res['jaccard'] = res[['answer', 'prediction']].apply(jaccard, axis=1)
res['postuned'] = res['prediction'].apply(postuning)
res['pjaccard'] = res[['answer', 'postuned']].apply(jaccard, axis=1)
res.jaccard.mean(), res.pjaccard.mean()


