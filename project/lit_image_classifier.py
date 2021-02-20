
import sys
sys.path.append(f'/users/Etu2/3701222/.local/lib/python3.7/site-packages')


from argparse import ArgumentParser
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

from vision_transformer_org import VisionTransformer

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

class LitClassifierModel(pl.LightningModule):

    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=False)
        self.log('train_acc', acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        acc = accuracy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max, verbose=True),
            'monitor': 'train_loss'
        }


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--t_max', type=float, default=100)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--num_workers', default=12, type=int)
        parser.add_argument('--image_size', default=512, type=int)
        parser.add_argument('--patch_size', default=4, type=int)
        parser.add_argument('--emb_dim', default=128, type=int)
        parser.add_argument('--mlp_dim', default=256, type=int)
        parser.add_argument('--num_heads', default=24, type=int)
        parser.add_argument('--num_layers', default=24, type=int)
        parser.add_argument('--num_classes', default=10, type=int)
        parser.add_argument('--attn_dropout_rate', default=0.0, type=float)
        parser.add_argument('--dropout_rate', default=.1, type=float)
        parser.add_argument('--resnet', action="store_true")
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument('--data_path', type=str)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifierModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    transform_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)

    trainloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size)
    testiloader = DataLoader(testset, num_workers=args.num_workers, batch_size=args.batch_size)

    # ------------
    # model
    # ------------

    print(len(dataset))

    vit_Backbone = VisionTransformer(
        image_size=(args.image_size, args.image_size),
        patch_size=(args.patch_size, args.patch_size),
        emb_dim=args.emb_dim,
        mlp_dim=args.mlp_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        attn_dropout_rate=args.attn_dropout_rate,
        dropout_rate=args.dropout_rate,
        resnet=args.resnet,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=f"{args.default_root_dir}/lightning_logs",
        filename='vit-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.resume_from_checkpoint:
        model = LitClassifierModel.load_from_checkpoint(checkpoint_path=args.resume_from_checkpoint)
    else:
        model = LitClassifierModel(vit_Backbone, **vars(args))
    pl.utilities.seed.seed_everything(123456789)
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor])

    if args.fit :
        trainer.fit(model, trainloader, testiloader)

    # ------------
    # testing
    result = trainer.test(model=model, test_dataloaders=testiloader)
    # ------------
    print(result)

if __name__ == '__main__':
    cli_main()
