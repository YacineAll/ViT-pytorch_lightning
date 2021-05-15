import argparse

import pytorch_lightning as pl
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from project.vision_transformer import VisionTransformer, Embedding_mode
from project.lightning_modules import CIFAR10DataModule, LitClassifierModel, load_from_checkpoint

def test_lit_classifier():
    seed_everything(12345)


    args = argparse.Namespace(attn_dropout_rate=0.0, batch_size=32, data_dir='/tmp', default_root_dir='/tmp', dropout_rate=0.1, emb_dim=32, embedding_mode=Embedding_mode.linear, fit=True, gpus=0, image_size=32, learning_rate=0.0001, max_epochs=1, mlp_dim=64, num_classes=10, num_heads=12, num_layers=12, num_workers=8, patch_size=4, progress_bar_refresh_rate=25, val_size=0.2, weight_decay=0.01)



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

    results = trainer.test(model=model, datamodule=datamodule)

    assert float(np.round(float(results[0]['acc']['avg']), 3)) == float(np.round(0.047623802, 3))