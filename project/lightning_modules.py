
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl


from argparse import ArgumentParser
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.datasets import CIFAR10
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tools import LabelSmoothingLoss

class LitClassifierModel(pl.LightningModule):

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs)
        self.criterion = LabelSmoothingLoss(kwargs['num_classes'], smoothing=0.2)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=False)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # loss = F.cross_entropy(logits, y)
        loss = self.criterion(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                min_lr=1e-8,
                verbose=True
            ),

            'monitor': 'train_loss'
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--patch_size', default=4, type=int)
        parser.add_argument('--emb_dim', default=128, type=int)
        parser.add_argument('--mlp_dim', default=256, type=int)
        parser.add_argument('--num_heads', default=24, type=int)
        parser.add_argument('--num_layers', default=24, type=int)
        parser.add_argument('--attn_dropout_rate', default=0.0, type=float)
        parser.add_argument('--dropout_rate', default=.1, type=float)
        parser.add_argument('--resnet', action="store_true")
        return parser


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', image_size: int = 512, batch_size: int = 128, num_workers: int = 12, val_size: float = 0.2):
        super().__init__()
        self.data_dir = data_dir

        self.transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.RandomRotation(2.8),
                transforms.GaussianBlur(kernel_size=11, sigma=(0.1, 2.0)),
            ]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size

        self.dims = (3, image_size, image_size)

    def prepare_data(self):
        # download
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            val_size = int(50000 * self.val_size)
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [50000 - val_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def load_from_checkpoint(model_class:type, lit_model:type, hparams_file: str, checkpoint_file:str):
    hparams = pl.core.saving.load_hparams_from_yaml(hparams_file)
    backbone = model_class(**hparams)
    model = lit_model.load_from_checkpoint(
                checkpoint_file,
                backbone=backbone
            )
    return model