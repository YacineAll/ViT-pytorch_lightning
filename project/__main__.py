import sys
import os
import argparse

import pytorch_lightning as pl

import numpy as np


from vision_transformer import VisionTransformer, Embedding_mode
from lightning_modules import CIFAR10DataModule, LitClassifierModel, load_from_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor



from pytorch_lightning.loggers import TestTubeLogger



def cli():
    pl.seed_everything(12345)

    parser = argparse.ArgumentParser()

    ##########useful args##########
    parser.add_argument('--fit', help="fit the model", action="store_true")
    parser.add_argument('--default_root_dir', default="./model", type=str, help="model directory")
    parser.add_argument('--data_dir', default="./", type=str, help="data directory")
    parser.add_argument('--gpus', default=-1, type=int, help="number of GPUs to use, -1 to all available gpus")
    ########Data Args##########
    parser.add_argument('--image_size', default=224, type=int, help="size of input images to the model")
    parser.add_argument('--num_classes', default=10, type=int, help="number of classes")
    ########Optimization Args##########
    parser.add_argument('--learning_rate', default=1e-4, type=float, help="learning rate for gradient descent")
    parser.add_argument('--weight_decay', default=1e-2, type=float, help="weight decay for model regularization")
    parser.add_argument('--attn_dropout_rate', default=0., type=float, help="dropout ratio for attention layers")
    parser.add_argument('--dropout_rate', default=0.1, type=float, help="dropout ratio")
    ########Training Args##########
    parser.add_argument('--val_size', default=0.2, type=float, help="the ratio of the validation dataset to the data set")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--num_workers', default=8, type=int, help="number of cpu workers to use")
    #######Model args############
    parser.add_argument('--patch_size', default=16, type=int, help="patch size model hyperparameters")
    parser.add_argument('--emb_dim', default=768, type=int, help="embedding dimension")
    parser.add_argument('--mlp_dim', default=3072, type=int, help="multilayer perceptron dimension")
    parser.add_argument('--num_heads', default=12, type=int, help="number of head for multi head attention")
    parser.add_argument('--num_layers', default=12, type=int, help="number of layers")
    parser.add_argument('--embedding_mode', default=Embedding_mode.linear, type=Embedding_mode, choices=list(Embedding_mode), help="embedding mode")
    ########Trainer Args##########
    parser.add_argument('--progress_bar_refresh_rate', default=25, type=int, help="prgress bar")
    parser.add_argument('--max_epochs', default=1, type=int, help="max iterations")



    args = parser.parse_args()



    datamodule = CIFAR10DataModule(**vars(args))

    vit_Backbone = VisionTransformer(**vars(args))

    checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            filename='vit-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
            mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')


    model = LitClassifierModel(vit_Backbone, **vars(args))

    logger = TestTubeLogger(args.default_root_dir, name='vit')
    trainer = pl.Trainer.from_argparse_args(args, accelerator='ddp', callbacks=[checkpoint_callback, lr_monitor], logger=[logger])

    trainer.fit(model,  datamodule)

    results = trainer.test(model=model, datamodule=datamodule)



if __name__ == '__main__':
    cli()
