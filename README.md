### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with CIFAR10

#### Goals
The objective of this project is to produce the ViT model with PyTorchLightning in a university project

---

<div align="center">    

# ViT Transformer PyTorchLightning

[![Paper](https://img.shields.io/badge/paper-arxiv.2010.11929-B31B1B.svg)](https://arxiv.org/pdf/2010.11929.pdf)
[![Conference](https://img.shields.io/badge/ICLR-2021-4b44ce.svg)](https://openreview.net/forum?id=YicbFdNTTy)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>


## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YacineAll/ViT-pytorch_lightning.git

# install project   
cd ViT-pytorch_lightning.git
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python __main__.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
import argparse

from project.vision_transformer import VisionTransformer, Embedding_mode
from project.lightning_modules import CIFAR10DataModule,LitClassifierModel, load_from_checkpoint

from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

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

trainer.fit(model,  datamodule)

results = trainer.test(model=model, datamodule=datamodule)



