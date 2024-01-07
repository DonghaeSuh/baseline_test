import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

import transformers
import pytorch_lightning as pl


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=256)

    def load_data(self, dataset):
        """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
        subject_entity = []
        object_entity = []
        for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
          i = i[1:-1].split(',')[0].split(':')[1]
          j = j[1:-1].split(',')[0].split(':')[1]

          subject_entity.append(i)
          object_entity.append(j)
        out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
        return out_dataset

    def tokenizing(self,dataset):
        """ tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []

        for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
          temp = ''
          temp = e01 + '[SEP]' + e02
          concat_entity.append(temp)

        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(dataset['sentence']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            )
        
        return tokenized_sentences

    def label_to_num(self,label):
        num_label = []
        with open('./utils/dict_label_to_num.pkl', 'rb') as f:
          dict_label_to_num = pickle.load(f)
        for v in label:
          num_label.append(dict_label_to_num.get(v,0))
        
        return num_label

    def preprocessing(self, data):

        # 텍스트 데이터를 전처리합니다.
        dataset = self.load_data(data)
        inputs = self.tokenizing(dataset)

        targets = self.label_to_num(dataset['label'].values)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = RE_Dataset(train_inputs, train_targets)
            self.val_dataset = RE_Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.dev_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = RE_Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = RE_Dataset(predict_inputs, predict_targets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)