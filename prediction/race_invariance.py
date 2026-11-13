import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from dotenv import load_dotenv
from typing import Literal
import torchvision
from tqdm import tqdm

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imsave
from argparse import ArgumentParser

import json

from prediction.backbones import DenseNet, ResNet18, ResNet34, ResNet50, ViTB16, EfficientNetB0
from prediction.datasets.cxr import CXRDataModule
from prediction.metrics import compute_metrics


load_dotenv()

num_classes_disease = 2
num_workers = 8


def resolve_accelerator(requested):
    """Resolve a requested execution device without silently falling back."""
    if requested == "auto":
        try:
            current = torch.accelerator.current_accelerator()
        except (AttributeError, RuntimeError):
            current = None
        if current is not None and getattr(current, "type", None) == "tpu":
            if not torch.accelerator.is_available():
                raise RuntimeError("TorchTPU is registered but no TPU device is available")
            return "tpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "tpu":
        try:
            import torch_tpu  # noqa: F401 - registers the native PyTorch TPU backend
        except ImportError as exc:
            raise RuntimeError(
                "TPU requested, but torch-tpu is not installed in this environment"
            ) from exc
        try:
            current = torch.accelerator.current_accelerator()
            available = torch.accelerator.is_available()
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError(
                "TPU requested, but this PyTorch build has no registered TPU backend"
            ) from exc
        if current is None or getattr(current, "type", None) != "tpu" or not available:
            raise RuntimeError("TPU requested, but no TorchTPU device is available")
        return "tpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available")
    return requested


def _move_batch_to_device(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _save_tpu_checkpoint(model, path, epoch, global_step, hparams):
    state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state_dict,
            "hyper_parameters": {
                "num_classes": num_classes_disease,
                "inv_loss_coefficient": hparams.inv_loss_coefficient,
            },
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def _validate_tpu(model, dataloader, device):
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            loss_class, _ = model.process_batch(batch)
            count = int(batch["label"].shape[0])
            weighted_loss += float(loss_class.detach().cpu()) * count
            sample_count += count
    if sample_count == 0:
        raise RuntimeError("Validation loader produced no samples")
    return weighted_loss / sample_count


def train_tpu(model, datamodule, logger, hparams):
    """Train on the single TPU using ordinary eager PyTorch, not Lightning/XLA."""
    device = torch.device("tpu")
    model.to(device)
    optimizer = model.configure_optimizers()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    log_dir = logger.log_dir
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best.ckpt")
    last_path = os.path.join(checkpoint_dir, "last.ckpt")
    best_val_loss = float("inf")
    global_step = 0
    max_steps = 5 if hparams.test else None
    val_interval = 2 if hparams.test else 300
    checkpoint_interval = 301
    last_validated_step = -1

    for epoch in range(hparams.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)
            invariant_batch = batch["image"].ndim == 5
            process_batch = model.process_batch_list if invariant_batch else model.process_batch
            optimizer.zero_grad(set_to_none=True)
            loss_class, loss_inv = process_batch(batch)
            loss = loss_class + loss_inv
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 5 == 0:
                logger.experiment.add_scalar(
                    "train_loss_class", float(loss_class.detach().cpu()), global_step
                )
                inv_value = (
                    float(loss_inv.detach().cpu())
                    if torch.is_tensor(loss_inv)
                    else float(loss_inv)
                )
                logger.experiment.add_scalar("train_loss_inv", inv_value, global_step)

            if global_step % checkpoint_interval == 0:
                _save_tpu_checkpoint(model, last_path, epoch, global_step, hparams)

            epoch_finished = batch_idx + 1 == len(train_loader)
            should_validate = global_step % val_interval == 0 or epoch_finished
            if should_validate and last_validated_step != global_step:
                val_loss = _validate_tpu(model, val_loader, device)
                logger.experiment.add_scalar("val_loss_class", val_loss, global_step)
                logger.experiment.add_scalar("val_loss_inv", 0.0, global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_tpu_checkpoint(model, best_path, epoch, global_step, hparams)
                last_validated_step = global_step
                model.train()

            if max_steps is not None and global_step >= max_steps:
                break

        if global_step > 0:
            _save_tpu_checkpoint(model, last_path, epoch, global_step, hparams)
        if max_steps is not None and global_step >= max_steps:
            break

    if hparams.test:
        return last_path
    if not os.path.exists(best_path):
        best_path = last_path
    return best_path


def datafiles(dataset: Literal["chexpert", "mimic"]):
    files = [
        f"datafiles/{dataset}/{dataset}.sample.{split}.csv"
        for split in ["train", "val", "test"]
    ]
    if dataset == "chexpert":
        files.append(os.getenv("CHEXPERT_FOLDER"))
    else:
        files.append(os.getenv("MIMIC_FOLDER"))
    return files


def test(model, data_loader, device, test_run):
    model.eval()
    logits_disease = []
    preds_disease = []
    embeddings = []
    targets_disease = []
    targets_invariant_attributes = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            if test_run and index == 3:
                break

            img, lab_disease, invariant_attribute = batch['image'].to(device), batch['label'].to(device), batch['invariant_attribute'].to(device)
            embedding, out_disease = model(img)

            pred_disease = torch.softmax(out_disease, dim=-1)

            logits_disease.append(out_disease)
            preds_disease.append(torch.argmax(pred_disease, -1))
            targets_disease.append(lab_disease)

            embeddings.append(embedding)
            targets_invariant_attributes.append(invariant_attribute)

        embeddings = torch.cat(embeddings, dim=0)
        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_invariant_attributes = torch.cat(targets_invariant_attributes, dim=0)

        info = {'global': {}, 'sub-group': {}}

        for i in range(0, num_classes_disease):
            t = targets_disease == 1
            c = torch.sum(t).item()
            info['global'][i] = int(c)

        for j in torch.unique(targets_invariant_attributes):
            info['sub-group'][int(j.item())] = {}
            for i in range(0, num_classes_disease):
                t = targets_disease[targets_invariant_attributes == j] == i
                c = torch.sum(t).item()
                info['sub-group'][int(j.item())][i] = int(c)

        print(json.dumps(info, indent=4))

    return (
        info,
        embeddings.cpu().numpy(),
        preds_disease.cpu().numpy(),
        targets_disease.cpu().numpy(),
        logits_disease.cpu().numpy(),
        targets_invariant_attributes.cpu().numpy(),
    )


def main(hparams):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(hparams.seed, workers=True)
    accelerator = resolve_accelerator(hparams.accelerator)
    if accelerator == "tpu" and hparams.gpus != 1:
        raise ValueError(
            "This training entry point currently supports one TPU device; set --gpus 1"
        )

    # model
    model_type = {
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "densenet": DenseNet,
        "vitb16": ViTB16,
        "efficientnetb0": EfficientNetB0,
    }[hparams.model_type]

    model = model_type(
        num_classes=num_classes_disease,
        inv_loss_coefficient=hparams.inv_loss_coefficient,
    )

    image_size = (hparams.size, hparams.size)

    # Create output directory
    logdir = "logs_test" if hparams.test else "logs"
    logdir = f'{logdir}/cxr_{hparams.invariance}_{hparams.disease.replace(" ", '_')}_{hparams.size}'

    # Setup datasets
    # Train set
    (
        csv_train_img,
        csv_val_img,
        csv_test_img,
        img_data_dir,
    ) = datafiles(hparams.dataset_train)
    if hparams.dataset_train == hparams.dataset_test:
        # Attribute_transfer - train with protected_race_set_train test with protected_race_set_test
        out_dir = f'{logdir}/{model_type.__name__}-{hparams.seed}/{hparams.dataset_train}/{hparams.protected_race_set_train}_{hparams.protected_race_set_test}'
       
        data_train = CXRDataModule(
            csv_train_img=csv_train_img,
            csv_val_img=csv_val_img,
            csv_test_img=csv_test_img,
            img_data_dir=img_data_dir,
            image_size=image_size,
            pseudo_rgb=True,
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_train,
            protected_label=hparams.invariance,
            disease_name=hparams.disease,
        )
        data_test = CXRDataModule(
            csv_train_img=csv_train_img,
            csv_val_img=csv_val_img,
            csv_test_img=csv_test_img,
            img_data_dir=img_data_dir,
            image_size=image_size,
            pseudo_rgb=True,
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_test,
            protected_label=hparams.invariance,
            disease_name=hparams.disease,
        )
    else:
        # Dataset transfer - train with data_train, test on data_test
        out_dir = f'{logdir}/{model_type.__name__}-{hparams.seed}/{hparams.dataset_train}_{hparams.dataset_test}'
       
        data_train = CXRDataModule(
            csv_train_img=csv_train_img,
            csv_val_img=csv_val_img,
            csv_test_img=csv_test_img,
            img_data_dir=img_data_dir,
            image_size=image_size,
            pseudo_rgb=True,
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_test,
            protected_label=hparams.invariance,
            disease_name=hparams.disease,
        )

        # Alternative test set 
        (
            test_csv_train_img,
            test_csv_val_img,
            test_csv_test_img,
            test_img_data_dir,
        ) = datafiles(hparams.dataset_test)
        data_test = CXRDataModule(
            csv_train_img=test_csv_train_img,
            csv_val_img=test_csv_val_img,
            csv_test_img=test_csv_test_img,
            image_size=image_size,
            img_data_dir=test_img_data_dir,
            pseudo_rgb=True,
            batch_size=hparams.batch_size,
            num_workers=num_workers,
            nsamples=hparams.nsamples,
            invariant_sampling=hparams.invariant_sampling,
            use_cache=False,
            protected_race_set_train=hparams.protected_race_set_train,
            protected_race_set_test=hparams.protected_race_set_test,
            protected_label=hparams.invariance,
            disease_name=hparams.disease,
        )

    # Set up logging dirs
    if hparams.invariant_sampling:
        out_dir = os.path.join(out_dir, f"invariant_nsamples_{hparams.nsamples}_{hparams.inv_loss_coefficient}")
    else:
        out_dir = os.path.join(out_dir, "non_invariant")
    os.makedirs(out_dir, exist_ok=True)

    # Lightning has no native TorchTPU accelerator integration. Use the
    # repository's Lightning path for CPU/CUDA and an eager PyTorch loop for TPU.
    logger = TensorBoardLogger(out_dir)
    trainer = None
    if accelerator == "tpu":
        log_dir = logger.log_dir
    else:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_class",
            mode="min",
            save_last=True,
            every_n_train_steps=0 if hparams.test else 301,
        )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            log_every_n_steps=5,
            max_epochs=hparams.epochs,
            accelerator="gpu" if accelerator == "cuda" else "cpu",
            devices=hparams.gpus,
            logger=logger,
            max_steps=5 if hparams.test else -1,
            val_check_interval=2 if hparams.test else 300,
        )
        trainer.logger._default_hp_metric = False
        log_dir = trainer.log_dir

    # Sample images
    samples_dir = os.path.join(log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    for idx in range(0, 5):
        sample = data_train.val_set.get_sample(idx)
        imsave(
            os.path.join(samples_dir, 'sample_' + str(idx) + '.jpg'),
            sample['image'].numpy().astype(np.uint8),
        )

    # Train
    if accelerator == "tpu":
        model_path = train_tpu(model, data_train, logger, hparams)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        device = torch.device("tpu")
    else:
        trainer.fit(model, data_train)
        if hparams.test:
            model_path = trainer.checkpoint_callback.last_model_path
        else:
            model_path = trainer.checkpoint_callback.best_model_path
        model = model_type.load_from_checkpoint(
            model_path,
            num_classes=num_classes_disease,
            inv_loss_coefficient=hparams.inv_loss_coefficient,
        )
        device = torch.device(
            "cuda:" + str(hparams.dev) if accelerator == "cuda" else "cpu"
        )

    model.to(device)

    # Compute metrics and embeddings and write to CSV
    def output_dfs(dataloader):
        (
            info,
            embeddings,
            preds_disease,
            targets_disease,
            logits_disease,
            target_invariant_attributes,
        ) = test(model, dataloader, device, hparams.test)
    
        cols_names_logits_disease = ['logit_' + str(i) for i in range(0, num_classes_disease)]

        df = pd.DataFrame(data=preds_disease, columns=["class"])
        df_logits = pd.DataFrame(data=logits_disease, columns=cols_names_logits_disease)
        df_targets = pd.DataFrame(data=targets_disease, columns=["target"])
        df_invariant_attributes = pd.DataFrame(data=target_invariant_attributes, columns=['Protected'])
        df_preds = pd.concat([df, df_logits, df_targets, df_invariant_attributes], axis=1)

        df_metrics = compute_metrics(df_preds)

        df_embed = pd.DataFrame(data=embeddings)
        df_embed = pd.concat([df_embed, df_targets, df_invariant_attributes], axis=1)
        
        return info, df_preds, df_embed, df_metrics

    dls = {
        "val": data_train.val_dataloader(),
        "test": data_train.test_dataloader(),
        "ood": data_test.test_dataloader(),
    }
    def write_dfs(split: Literal["val", "test", "ood"]):
        metrics_dir = os.path.join(log_dir, "output")
        os.makedirs(metrics_dir, exist_ok=True)
        dl = dls[split]
        info, df_preds, df_embedding, df_metrics = output_dfs(dl)
        with open(os.path.join(metrics_dir, f'count_info_{split}.json'), 'w') as f:
            json.dump(info, f, indent=4)
        df_preds.to_csv(os.path.join(metrics_dir, f'predictions.{split}.csv'), index=False)
        df_embedding.to_csv(os.path.join(metrics_dir, f'embeddings.{split}.csv'), index=False)
        df_metrics.to_csv(os.path.join(metrics_dir, f'metrics.{split}.csv'), index=False)

    write_dfs("val")
    write_dfs("test")
    write_dfs("ood")
    if accelerator == "tpu":
        logger.experiment.close()


def cli():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of CPU/CUDA devices; TPU mode currently requires 1')
    parser.add_argument('--dev', type=int, default=0)
    parser.add_argument('--accelerator', choices=['auto', 'cpu', 'cuda', 'tpu'],
                        default='auto',
                        help='auto prefers an available TorchTPU device, then CUDA, then CPU')
    parser.add_argument('--test', action="store_true")

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--nsamples', type=int, default=1)
    parser.add_argument('--inv-loss-coefficient', type=float, default=1)
    parser.add_argument('--invariant-sampling', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--protected-race-set-train', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--protected-race-set-test', nargs='+', type=int, default=[0, 1, 2, 3])

    parser.add_argument('--dataset-train', choices=["mimic", "chexpert"], default="chexpert")
    parser.add_argument('--dataset-test', choices=["mimic", "chexpert"], default="chexpert")
    parser.add_argument(
        '--model-type',
        choices=["densenet", "resnet34", "resnet50", "vitb16", "resnet18", "efficientnetb0"],
        default="densenet",
    )

    parser.add_argument('--disease', choices=["Cardiomegaly", "Pleural Effusion"], required=True)
    parser.add_argument('--invariance', choices=["race", "view", "sex"], required=True)

    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    cli()
