import argparse
import pandas as pd
import os
import re
import time
from tqdm import tqdm
import numpy as np
import csv
import sys
import random
import logging
import urllib3
import json
import pickle
import itertools
import openpyxl

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, RobertaForSequenceClassification, AutoConfig

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertForSequenceClassificationMLP1
from pytorch_pretrained_bert.tokenization_morp import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class PLM_Processor():
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir, sheet_num, guid_s_num):
        """See base class."""
        return self._create_examples(
            self._read_excel(data_dir, sheet_num), sheet_num, guid_s_num,
            "train")

    def get_dev_examples(self, data_dir, sheet_num, guid_s_num):
        """See base class."""
        return self._create_examples(
            self._read_excel(data_dir, sheet_num), sheet_num, guid_s_num,
            "dev")

    def get_labels(self, data_dir):
        """See base class."""
        ######################################################################
        ### kyoungman.bae @ 19-05-30 @ for multi label classification
        ### You need to add a file with label information in the data folder.
        ### You should use a numbered label on a line.
        labels = []
        lines = self._read_tsv(os.path.join(data_dir, "labels.tsv"))
        for (i, line) in enumerate(lines):
            labels.append(str(line[0]))
        return labels

    def _read_excel(self, input_file, sheet_name):
        """Reads a tab separated value file."""
        lines = pd.read_excel(input_file, sheet_name=str(sheet_name))
        for i in range(len(lines)):
            lines['Script'][i] = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ·!』\\‘’|\(\)\[\]\<\>`\'…》]', '', lines['Script'][i])
        print(lines['Script'])
        return lines

    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='UTF8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(np.unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _create_examples(self, lines, sheet_num, guid_s_num, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        full_text = lines['Script']
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, guid_s_num + i)
            text_a = full_text[i]
            label = str(sheet_num)  # 이것이 label 설정
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def do_lang(openapi_key, text):
    # 문어체
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"

    # 구어체
    # openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU_spoken"

    requestJson = {"access_key": openapi_key, "argument": {"text": text, "analysis_code": "morp"}}

    http = urllib3.PoolManager()
    response = http.request("POST", openApiURL, headers={"Content-Type": "application/json; charset=UTF-8"},
                            body=json.dumps(requestJson))

    json_data = json.loads(response.data.decode('utf-8'))
    json_result = json_data["result"]

    if json_result == -1:
        json_reason = json_data["reason"]
        if "Invalid Access Key" in json_reason:
            logger.info(json_reason)
            logger.info("Please check the openapi access key.")
            sys.exit()
        return "openapi error - " + json_reason
    else:
        json_data = json.loads(response.data.decode('utf-8'))

        json_return_obj = json_data["return_object"]

        return_result = ""
        json_sentence = json_return_obj["sentence"]
        for json_morp in json_sentence:
            for morp in json_morp["morp"]:
                return_result = return_result + str(morp["lemma"]) + "/" + str(morp["type"]) + " "

        return return_result


class PLM_Dataset(Dataset):
    def __init__(self, text_tuple, tokenizer=None, max_length=None, is_label=False, openapi_key=None):
        self.text_tuple = text_tuple
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_label = is_label
        self.openapi_key = openapi_key

    def __len__(self):
        return len(self.text_tuple)

    def __getitem__(self, idx):
        text = self.text_tuple[idx].text_a

        if self.openapi_key:
            text = do_lang(self.openapi_key, text)

        token_list = self.tokenizer.tokenize(text)
        if len(token_list) > 254:
            token_list = token_list[0:254]

        # token -> ids 변경
        token_list = ["[CLS]"] + token_list + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        input_mask = [1] * len(input_ids)

        # padding
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        if self.is_label:
            label = int(self.text_tuple[idx].label)
            return tuple([input_ids, input_mask, label])
        else:
            return tuple([input_ids, input_mask])


class KorBERT_Dataset(Dataset):
    def __init__(self, text_tuple, tokenizer=None, max_length=None, is_label=False, openapi_key=None):
        self.text_tuple = text_tuple
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_label = is_label
        self.openapi_key = openapi_key

    def __len__(self):
        return len(self.text_tuple)

    def __getitem__(self, idx):
        text = self.text_tuple[idx]['tokens']
        # 형님이 주신 코드에서 추가해야할 부분, korbert tokenizer는 max_length 128 넘어가는 test data 존재함 (train 데이터는 안그랬는데...)
        if len(text) > 254:
            text = text[0:254]

        # token -> ids 변경
        token_list = ["[CLS]"] + text + ["[SEP]"]
        segment_ids = [0] * len(token_list)

        input_ids = self.tokenizer.convert_tokens_to_ids(token_list)
        input_mask = [1] * len(input_ids)

        # padding
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_length
        assert len(input_mask) == self.max_length
        assert len(segment_ids) == self.max_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_label:
            label = int(self.text_tuple[idx]['label'])
            return tuple([input_ids, input_mask, segment_ids, label])
        else:
            return tuple([input_ids, input_mask, segment_ids])

def save_confusion_matrix(y_label, y_pred, model_name, save_dir, normalize = 'true'):
    font_path = "C:/Windows/Fonts/gulim.ttc"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

    cf = confusion_matrix(y_label, y_pred, normalize= normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=['비폭력','협박','갈취','직장 내 괴롭힘','기타 괴롭힘'])
    disp.plot(cmap='Reds')
    plt.xticks(rotation=20)
    plt.title(model_name)
    plt.savefig(save_dir+model_name+ '.png')

def main(args, model_name_list, augmentation):
    # gpu count
    n_gpu = torch.cuda.device_count()

    # random seed 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # GPU 사용
    device = torch.device("cuda:0")

    if args.is_ensemble_test:
        ensemble_y_pred={} # ensemble의 결과를 모두 담기 위해 거의 전역변수 위치로 dictionary 선언

    for model_name in model_name_list:
        openapi_key = None
        if len(model_name.split('/')) == 1:
            # KorBERT 사용 하는 경우
            if model_name == "KorBERT":
                BertForSequenceClassification = BertForSequenceClassificationMLP1
                model_path = os.path.join(args.model_dir, model_name)

                # tokenizer 및 모델 선언8
                tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "vocab.korean_morp.list"), do_lower_case=False)
                model = BertForSequenceClassification.from_pretrained(model_path, num_labels=args.num_labels).to(device)

                # ETRI open api key load
                with open(os.path.join(model_path, args.openapi_key_name), 'r') as f:
                    openapi_key_list = f.read().split()
                openapi_key = openapi_key_list[0]

            # ELECTRA 기반 NeuroAI 모델 사용하는 경우
            else:
                model_path = os.path.join(args.model_dir, model_name)
                config = AutoConfig.from_pretrained(model_path, num_labels=args.num_labels)

                # tokenizer 및 모델 선언
                tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
                model = ElectraForSequenceClassification.from_pretrained(model_path, config=config).to(device)
        # Huggingface를 통해 배포된 한국어 기반 PLM 사용하는 경우
        else:
            # tokenizer 및 모델 선언
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if 'klue/roberta' in model_name:
                model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels).to(
                    device)
            else:
                model = ElectraForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels).to(
                    device)

        # KorBERT tokenized feature load and define dataset version
        if model_name == "KorBERT":
            dataset = KorBERT_Dataset
            with open(os.path.join(args.feature_dir, "train_text.pickle"), "rb") as fr:
                train_tokens = pickle.load(fr)

            with open(os.path.join(args.feature_dir, "test_total_list.pickle"), "rb") as fr:
                eval_tokens = pickle.load(fr)

        else:
            dataset = PLM_Dataset
            # data load
            # train data load
            total_examples, file_name, processor = [], args.train_data_name, PLM_Processor()
            for s_num in range(args.sheet_num):
                total_examples.extend(processor.get_train_examples(args.data_dir + file_name,
                                                                   s_num, len(total_examples)))
            train_tokens = total_examples

            # test data load
            test_examples, file_name, processor = [], args.test_data_name, PLM_Processor()
            for s_num in range(args.sheet_num):
                test_examples.extend(processor.get_train_examples(args.data_dir + file_name,
                                                                  s_num, len(test_examples)))
            eval_tokens = test_examples

            del total_examples
            del test_examples

        # augmentation
        if augmentation:
            if model_name == "KorBERT":
                with open(os.path.join(args.feature_dir, "train_augmentation_list.pickle"), "rb") as fr:
                    aug_total_examples = pickle.load(fr)
            else:
                aug_total_examples, file_name, processor = [], args.augmentation_train_data_name, PLM_Processor()
                for s_num in range(args.sheet_num):
                    aug_total_examples.extend(
                        processor.get_train_examples(args.data_dir + file_name, s_num, len(aug_total_examples)))

            for aug_ratio in augmentation:

                for class_num in range(0, 5):
                    class_nums = random.sample(range(args.augmentation_data_class_num * class_num,
                                                     args.augmentation_data_class_num * (class_num + 1)),
                                               int(args.augmentation_data_class_num * aug_ratio))

                    for idx in range(args.augmentation_data_class_num * class_num,
                                                     args.augmentation_data_class_num * (class_num + 1)):
                        if idx in class_nums:
                            train_tokens.append(aug_total_examples[idx])



                # train, test Dataset 구성
                train_dataset = dataset(train_tokens, tokenizer, args.max_length, is_label=True, openapi_key=openapi_key)
                test_dataset = dataset(eval_tokens, tokenizer, args.max_length, is_label=True, openapi_key=openapi_key)

                # train, test Dataloader 구성
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                # Prepare optimizer
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

                # optimizer 설정
                num_train_optimization_steps = int(len(train_tokens) / args.batch_size) * args.epochs

                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

                # training loop
                if args.is_training:
                    losses, accuracies = [], []
                    for i in range(args.epochs):
                        total_loss, correct, total = 0.0, 0, 0
                        model.train()
                        for step, batch in enumerate(tqdm(train_loader)):
                            optimizer.zero_grad()

                            batch = tuple(t.to(device) for t in batch)
                            if model_name == "KorBERT":
                                input_ids_batch, attention_masks_batch, segment_ids_batch, y_batch = batch
                                y_pred = model(input_ids_batch, segment_ids_batch, attention_masks_batch)
                            else:
                                input_ids_batch, attention_masks_batch, y_batch = batch
                                y_pred = model(input_ids_batch, attention_mask=attention_masks_batch)[0]
                            loss = F.cross_entropy(y_pred, y_batch)

                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()

                            _, predicted = torch.max(y_pred, 1)
                            batch_correct = (predicted == y_batch).sum()
                            correct += batch_correct
                            total += len(y_batch)

                            if step != 0 and step % 50 == 0:
                                logger.info("Batch Loss: %f / Batch Accuracy: %f", loss.item(), batch_correct.float() / args.batch_size)

                        losses.append(total_loss / (step + 1))
                        accuracies.append(correct.float() / total)
                        logger.info("Train Loss: %f / Train Accuracy: %f", total_loss / (step + 1), correct.float() / total)

                    # 모델 저장하기
                    if len(model_name.split('/')) == 1:
                        torch.save(model.state_dict(), os.path.join(args.model_save_dir, model_name + "-" + str(aug_ratio) + "_model.pt"))
                        logger.info(model_name + "-" + str(aug_ratio)+" : save complete!")
                    else:
                        torch.save(model.state_dict(), os.path.join(args.model_save_dir, model_name.split("/")[1] + "-" + str(aug_ratio) + "_model.pt"))
                        logger.info(model_name.split("/")[1] + "-" + str(aug_ratio) + " : save complete!")

                if args.is_test:
                    # 여기에 directory안의 pt 파일 수가 ensemble_comb_n 보다 작으면 assert error
                    pt_list = [file for file in os.listdir(args.model_ensemble_dir) if file.endswith('pt')]
                    # assert len(pt_list) >= args.ensemble_comb_n

                    start_time = time.time()

                    if len(model_name.split('/')) == 1:
                        if args.is_ensemble_test:
                            if os.path.exists(os.path.join(args.model_ensemble_dir, model_name + "-" + str(aug_ratio) + "_model.pt")):
                                model.load_state_dict(torch.load(os.path.join(args.model_ensemble_dir, model_name + "-" + str(aug_ratio) + "_model.pt")))
                            else:
                                continue

                        else:
                            model.load_state_dict(torch.load(
                                os.path.join(args.model_save_dir, model_name + "-" + str(aug_ratio) + "_model.pt")))

                    else:
                        if args.is_ensemble_test:
                            if os.path.exists(os.path.join(args.model_ensemble_dir, model_name.split("/")[1] + "-" + str(aug_ratio) + "_model.pt")):
                                model.load_state_dict(torch.load(os.path.join(args.model_ensemble_dir, model_name.split("/")[1] + "-" + str(aug_ratio) + "_model.pt")))
                            else:
                                continue
                        else:
                            model.load_state_dict(torch.load(os.path.join(args.model_save_dir, model_name.split("/")[1] + "-" + str(aug_ratio) + "_model.pt")))

                    # test loop
                    model.eval()
                    if args.is_ensemble_test:
                        y_pred = []
                        for step, batch in enumerate(tqdm(test_loader)):
                            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                            if model_name == "KorBERT":
                                input_ids_batch, attention_masks_batch, segment_ids_batch, y_batch = batch
                                y_pred.append(model(input_ids_batch, segment_ids_batch, attention_masks_batch)[0].tolist())

                            else:
                                input_ids_batch, attention_masks_batch, y_batch = batch
                                y_pred.append(model(input_ids_batch, attention_mask=attention_masks_batch)[0][0].tolist())

                        if len(model_name.split('/')) == 1:
                            model_key_name = model_name + "-" + str(aug_ratio)
                        else:
                            model_key_name = model_name.split("/")[1] + "-" + str(aug_ratio)

                        ensemble_y_pred[model_key_name] = y_pred

                    else:
                        correct, total = 0, 0
                        all_pred_out, all_label_ids = [], []
                        for step, batch in enumerate(tqdm(test_loader)):
                            batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                            if model_name == "KorBERT":
                                input_ids_batch, attention_masks_batch, segment_ids_batch, y_batch = batch
                                y_pred = model(input_ids_batch, segment_ids_batch, attention_masks_batch)
                            else:
                                input_ids_batch, attention_masks_batch, y_batch = batch
                                y_pred = model(input_ids_batch, attention_mask=attention_masks_batch)[0]

                            _, predicted = torch.max(y_pred, 1)

                            all_pred_out.append(predicted.item())
                            all_label_ids.append(y_batch.item())

                            batch_correct = (predicted == y_batch.item()).sum()
                            correct += batch_correct
                            total += len(y_batch)

                        if len(model_name.split('/')) == 1:
                            model_key_name = model_name + "-" + str(aug_ratio)
                        else:
                            model_key_name = model_name.split("/")[1] + "-" + str(aug_ratio)
                        save_confusion_matrix(all_label_ids, all_pred_out, model_key_name, args.confusion_dir)

                        eval_f1_score = f1_score(all_label_ids, all_pred_out, average='macro')
                        logger.info(model_name + "-" + str(aug_ratio) + " [result]")
                        logger.info("Test Accuracy: %f", correct.float() / total)
                        logger.info("Test Macro F1 score: %f ", eval_f1_score)
                        total_time = time.time() - start_time
                        minn = int(total_time / 60)
                        sec = int(total_time % 60)
                        print("추출 소모 시간:", minn, ":", sec)

    if args.is_ensemble_test: # 여기서 데이터 정리하고 위에서 저장한 데이터들 조합으로 결과들 낼 수 있도록 하기 (sklearn 조합)이라고 검색해보기

        ensemble_comb = list(itertools.combinations((ensemble_y_pred),args.ensemble_comb_n))
        for comb_tuple in ensemble_comb:
            ensemble_sum = []
            for model_key in comb_tuple:
                if ensemble_sum: # ensemble_y_pred라는 dictionary에 'sum' 이라는 key 값이 존재하는지 확인 하는 부분
                    for idx, (ensemble_y, y) in enumerate(zip(ensemble_sum, ensemble_y_pred.get(model_key))):
                        ensemble_sum[idx] = [x + y for x, y in zip(ensemble_y, y)]  # 여기서 덧셈 진행하는데 여기서 그냥 저장 결과를 특정 리스트에 저장되도록만 하기
                else:
                    for i in ensemble_y_pred[model_key]:
                        ensemble_sum.append(i)
                    # ensemble_y_pred['sum'] = ensemble_y_pred.get(model_key)

            correct, total = 0, 0
            all_pred_out, all_label_ids = [], []
            for step, batch in enumerate(tqdm(test_loader)):
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                _, _, y_batch = batch

                predicted = np.argmax(np.array(ensemble_sum[step]))

                all_pred_out.append(predicted)
                all_label_ids.append(y_batch.item())

                batch_correct = (predicted == y_batch.item()).sum()
                correct += batch_correct
                total += len(y_batch)

            save_confusion_matrix(all_label_ids, all_pred_out, str(comb_tuple)[1:len(str(comb_tuple))-1], args.confusion_dir)

            eval_f1_score = f1_score(all_label_ids, all_pred_out, average='macro')

            logger.info("Ensemble combination : %s", str(comb_tuple))
            logger.info("Ensemble Test Accuracy: %f ", correct / total)
            logger.info("Ensemble Test Macro F1 score: %f ", eval_f1_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--augmentation_data_total_num", type=int, default=5000)

    parser.add_argument("--test_count", type=int, default=1000)
    parser.add_argument("--data_total_num", type=int, default=5005)
    parser.add_argument("--augmentation_data_class_num", type=int, default=1000)

    parser.add_argument("--num_labels", type=int, default=5)
    parser.add_argument("--sheet_num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)

    parser.add_argument("--model_dir", type=str, default='./model')
    parser.add_argument("--model_save_dir", type=str, default='./save_models')
    parser.add_argument("--model_ensemble_dir", type=str, default='./ensemble_models')
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--feature_dir", type=str, default='./features/')
    parser.add_argument("--confusion_dir", type=str, default='./confusion_mtx/')

    parser.add_argument("--train_data_name", type=str, default='train_text.xlsx')
    parser.add_argument("--augmentation_train_data_name", type=str, default='train_augmentation_list.xlsx')
    parser.add_argument("--test_data_name", type=str, default='test_list.xlsx')
    parser.add_argument("--openapi_key_name", type=str, default='openapi_key_list.txt')

    parser.add_argument("--is_training", type=bool, default=True)
    parser.add_argument("--is_test", type=bool, default=True)
    parser.add_argument("--is_ensemble_test", type=bool, default=False)
    parser.add_argument("--ensemble_comb_n", type=int, default=3)

    args = parser.parse_args()

    model_name_list = [
                       "beomi/KcELECTRA-base",
                       "KorBERT",
                       "tunib/electra-ko-base",
                       "monologg/koelectra-base-discriminator",
                       "monologg/koelectra-base-v2-discriminator",
                       "monologg/koelectra-base-v3-discriminator",
                       "KR_ELECTRA_base_mecab",
                       ]

    augmentation = [0, 0.25, 0.5, 0.75, 1.0]

    main(args, model_name_list, augmentation)
