import argparse
import random
from tqdm.auto import tqdm
from typing import Dict
import json
import importlib

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
import torch

import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb
from pytorch_lightning.loggers import WandbLogger

from dataloader import *
from models import base_model

MODEL = base_model


def main(config: Dict):
    #seed 고정
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=config['arch']['model_name'], type=str)
    parser.add_argument('--model_detail', default=config['arch']['model_detail'], type=str)

    parser.add_argument('--batch_size', default=config['trainer']['batch_size'], type=int)
    parser.add_argument('--max_epoch', default=config['trainer']['max_epoch'], type=int)
    parser.add_argument('--shuffle', default=config['trainer']['shuffle'], type=bool)
    parser.add_argument('--learning_rate', default=config['trainer']['learning_rate'], type=float)

    parser.add_argument('--train_path', default=config['path']['train_path'], type=str)
    parser.add_argument('--dev_path', default=config['path']['dev_path'], type=str)
    parser.add_argument('--test_path', default=config['path']['test_path'], type=str)
    parser.add_argument('--predict_path', default=config['path']['predict_path'], type=str)
    args = parser.parse_args()

    wandb_logger = WandbLogger(name=config['wandb']['wandb_run_name'], project=config['wandb']['wandb_project_name'], entity=config['wandb']['wandb_entity_name']) # config로 설정 관리

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = getattr(MODEL,config['arch']['selected_model'])(args.model_name, args.learning_rate)



    early_stop_custom_callback = EarlyStopping(
        "val micro f1 score", patience=3, verbose=True, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val micro f1 score",
        save_top_k=1,
        dirpath="./",
        filename='./best_model/'+'_'.join(args.model_name.split('/') + args.model_detail.split()), # model에 따라 변화
        save_weights_only=False,
        verbose=True,
        mode="max",
    )


    # gpu가 없으면 accelerator="cpu"로 변경해주세요, gpu가 여러개면 'devices=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=args.max_epoch, callbacks=[checkpoint_callback,early_stop_custom_callback],log_every_n_steps=1,logger=wandb_logger)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)

    model = getattr(MODEL,config['arch']['selected_model'])(args.model_name, args.learning_rate)
    filename='./best_model/'+'_'.join(args.model_name.split('/') + args.model_detail.split()) + '.ckpt'
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, './best_model/'+'_'.join(args.model_name.split('/') + args.model_detail.split()) + '.pt')


if __name__ == '__main__':

    selected_config = 'base_config.json'

    with open(f'./configs/{selected_config}', 'r') as f:
        config = json.load(f)

    main(config=config)
