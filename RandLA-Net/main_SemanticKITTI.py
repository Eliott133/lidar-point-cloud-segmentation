import os
import time
import yaml
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from helper_tool import ConfigSemanticKITTI as cfg
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from semantic_kitti_dataset import SemanticKITTI


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------
# 1) Parser minimal pour récupérer le chemin du fichier de config
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=os.path.join(BASE_DIR, "configs", "config_randlanet.yaml"),
    help="Chemin vers le fichier YAML de configuration",
)

args_config, remaining_argv = parser.parse_known_args()

# ----------------------------------------------------------------------
# 2) Chargement du YAML
# ----------------------------------------------------------------------
yaml_cfg = load_yaml_config(args_config.config)

# ----------------------------------------------------------------------
# 3) Parser complet : valeurs par défaut venant du YAML
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default=args_config.config,
    help="Chemin vers le fichier YAML de configuration",
)
parser.add_argument(
    "--yaml_config",
    type=str,
    default=yaml_cfg.get("yaml_config", os.path.join(BASE_DIR, "semantic-kitti.yaml")),
    help="Chemin vers le fichier semantic-kitti.yaml",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=yaml_cfg.get("checkpoint_path", "output/checkpoint.tar"),
    help="Chemin du checkpoint du modèle",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=yaml_cfg.get("log_dir", "output"),
    help="Dossier de sortie pour les logs et checkpoints",
)
parser.add_argument(
    "--max_epoch",
    type=int,
    default=yaml_cfg.get("max_epoch", 400),
    help="Nombre maximal d'epochs",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=yaml_cfg.get("batch_size", 20),
    help="Taille du batch",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=yaml_cfg.get("num_workers", 4),
    help="Nombre de workers du DataLoader",
)

FLAGS = parser.parse_args()

# ----------------------------------------------------------------------
# 4) Logs
# ----------------------------------------------------------------------
LOG_DIR = FLAGS.log_dir
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "a")


def log_string(out_str: str) -> None:
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


# ----------------------------------------------------------------------
# 5) Dataset
# ----------------------------------------------------------------------
def my_worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


TRAIN_DATASET = SemanticKITTI("training")
TEST_DATASET = SemanticKITTI("validation")

TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET,
    batch_size=FLAGS.batch_size,
    shuffle=True,
    num_workers=FLAGS.num_workers,
    worker_init_fn=my_worker_init_fn,
    collate_fn=TRAIN_DATASET.collate_fn,
)

TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=FLAGS.batch_size,
    shuffle=True,
    num_workers=FLAGS.num_workers,
    worker_init_fn=my_worker_init_fn,
    collate_fn=TEST_DATASET.collate_fn,
)

log_string(f"Train samples: {len(TRAIN_DATASET)}")
log_string(f"Val samples: {len(TEST_DATASET)}")
log_string(f"Train batches: {len(TRAIN_DATALOADER)}")
log_string(f"Val batches: {len(TEST_DATALOADER)}")
log_string(f"Batch size: {FLAGS.batch_size}")
log_string(f"Num workers: {FLAGS.num_workers}")
log_string(f"Config file: {FLAGS.config}")
log_string(f"SemanticKITTI YAML: {FLAGS.yaml_config}")

# ----------------------------------------------------------------------
# 6) Réseau
# ----------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Network(cfg)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

start_epoch = 0
CHECKPOINT_PATH = FLAGS.checkpoint_path

if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    log_string(f"-> loaded checkpoint {CHECKPOINT_PATH} (epoch: {start_epoch})")

if torch.cuda.device_count() > 1:
    log_string(f"Let's use {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)


# ----------------------------------------------------------------------
# 7) Fonctions d'entraînement
# ----------------------------------------------------------------------
def adjust_learning_rate(optimizer, epoch: int) -> None:
    lr = optimizer.param_groups[0]["lr"]
    lr = lr * cfg.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch() -> None:
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    current_lr = optimizer.param_groups[0]["lr"]
    log_string(f"[TRAIN] Epoch {EPOCH_CNT:03d} started | lr={current_lr:.8f}")

    net.train()
    iou_calc = IoUCalculator(cfg)

    epoch_start_time = time.time()
    batch_start_time = time.time()

    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        for key in batch_data:
            if isinstance(batch_data[key], list):
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        optimizer.zero_grad()
        end_points = net(batch_data)

        loss, end_points = compute_loss(end_points, cfg)
        loss.backward()
        optimizer.step()

        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)

        for key in end_points:
            if "loss" in key or "acc" in key or "iou" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            elapsed = time.time() - batch_start_time
            progress = 100.0 * (batch_idx + 1) / len(TRAIN_DATALOADER)

            log_string(
                f"[TRAIN] Epoch {EPOCH_CNT:03d} | "
                f"batch {batch_idx + 1:04d}/{len(TRAIN_DATALOADER):04d} "
                f"({progress:5.1f}%) | "
                f"{elapsed / batch_interval:.3f}s/batch"
            )

            for key in sorted(stat_dict.keys()):
                log_string(f"[TRAIN] mean {key}: {stat_dict[key] / batch_interval:.6f}")
                stat_dict[key] = 0

            batch_start_time = time.time()

    mean_iou, iou_list = iou_calc.compute_iou()
    epoch_time = time.time() - epoch_start_time

    log_string(f"[TRAIN] Epoch {EPOCH_CNT:03d} finished in {epoch_time / 60:.2f} min")
    log_string("[TRAIN] mean IoU:{:.1f}".format(mean_iou * 100))
    s = "[TRAIN] IoU: "
    for iou_tmp in iou_list:
        s += "{:5.2f} ".format(100 * iou_tmp)
    log_string(s)


def evaluate_one_epoch() -> None:
    stat_dict = {}
    net.eval()
    iou_calc = IoUCalculator(cfg)

    log_string(f"[VAL] Epoch {EPOCH_CNT:03d} started")
    val_start_time = time.time()

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if isinstance(batch_data[key], list):
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        with torch.no_grad():
            end_points = net(batch_data)

        loss, end_points = compute_loss(end_points, cfg)
        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)

        for key in end_points:
            if "loss" in key or "acc" in key or "iou" in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % 10 == 0:
            progress = 100.0 * (batch_idx + 1) / len(TEST_DATALOADER)
            log_string(
                f"[VAL] Epoch {EPOCH_CNT:03d} | "
                f"batch {batch_idx + 1:04d}/{len(TEST_DATALOADER):04d} "
                f"({progress:5.1f}%)"
            )

    for key in sorted(stat_dict.keys()):
        log_string(f"[VAL] mean {key}: {stat_dict[key] / float(batch_idx + 1):.6f}")

    mean_iou, iou_list = iou_calc.compute_iou()
    val_time = time.time() - val_start_time

    log_string(f"[VAL] Epoch {EPOCH_CNT:03d} finished in {val_time / 60:.2f} min")
    log_string("[VAL] mean IoU:{:.1f}".format(mean_iou * 100))
    s = "[VAL] IoU: "
    for iou_tmp in iou_list:
        s += "{:5.2f} ".format(100 * iou_tmp)
    log_string(s)


def train(start_epoch: int) -> None:
    global EPOCH_CNT
    loss = 0

    log_string("====================================")
    log_string("START TRAINING")
    log_string(f"start_epoch: {start_epoch}")
    log_string(f"max_epoch: {FLAGS.max_epoch}")
    log_string(f"log_dir: {LOG_DIR}")
    log_string(f"checkpoint_path: {FLAGS.checkpoint_path}")
    log_string("====================================")

    for epoch in range(start_epoch, FLAGS.max_epoch):
        EPOCH_CNT = epoch
        log_string(f"**** EPOCH {epoch:03d} / {FLAGS.max_epoch - 1:03d} ****")
        log_string(str(datetime.now()))

        np.random.seed()
        train_one_epoch()

        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9:
            log_string(f"**** EVAL EPOCH {epoch:03d} START ****")
            evaluate_one_epoch()
            log_string(f"**** EVAL EPOCH {epoch:03d} END ****")

        save_dict = {
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }

        try:
            save_dict["model_state_dict"] = net.module.state_dict()
        except Exception:
            save_dict["model_state_dict"] = net.state_dict()

        ckpt_path = os.path.join(LOG_DIR, "checkpoint.tar")
        torch.save(save_dict, ckpt_path)
        log_string(f"[CKPT] Saved checkpoint to {ckpt_path} (next epoch: {epoch + 1})")


if __name__ == "__main__":
    train(start_epoch)