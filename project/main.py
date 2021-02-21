import sys

import pytorch_lightning as pl

from argparse import ArgumentParser
from vision_transformer_org import VisionTransformer
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_modules import CIFAR10DataModule, LitClassifierModel, load_from_checkpoint


def cli_main():
    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--num_workers', default=12, type=int)

    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitClassifierModel.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    datamodule = CIFAR10DataModule(
        data_dir=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_size=args.val_size
    )

    # ------------
    # model
    # ------------
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
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        filename='vit-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, lr_monitor])

    if args.resume_from_checkpoint:
        model = load_from_checkpoint(
            model_class=VisionTransformer,
            lit_model=LitClassifierModel,
            hparams_file=f"{trainer.logger.log_dir}/hparams.yaml",
            checkpoint_file=f"{trainer.logger.log_dir}/checkpoints/{args.resume_from_checkpoint}"
        )
        # model = LitClassifierModel.load_from_checkpoint(checkpoint_path=args.resume_from_checkpoint)
    else:
        model = LitClassifierModel(vit_Backbone, **vars(args))

    if args.fit:
        trainer.fit(model, datamodule)

    # ------------
    # testing
    # ------------
    result = trainer.test(model=model, datamodule=datamodule)

    print(result)


if __name__ == '__main__':
    sys.path.append(f'/users/Etu2/3701222/.local/lib/python3.7/site-packages')
    cli_main()
