from helper_tool import ConfigSemanticKITTI as cfg
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from semantic_kitti_dataset import SemanticKITTI
import numpy as np
import os, argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='output/checkpoint.tar', help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 8]')
FLAGS = parser.parse_args()

#################################################   log   #################################################
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


#################################################   dataset   #################################################
# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
TRAIN_DATASET = SemanticKITTI('training')
TEST_DATASET = SemanticKITTI('validation')
print(len(TRAIN_DATASET), len(TEST_DATASET))
NUM_WORKERS = 4
TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET,
    batch_size=FLAGS.batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=my_worker_init_fn,
    collate_fn=TRAIN_DATASET.collate_fn
)
TEST_DATALOADER = DataLoader(
    TEST_DATASET,
    batch_size=FLAGS.batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    worker_init_fn=my_worker_init_fn,
    collate_fn=TEST_DATASET.collate_fn
)

print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

log_string(f"Train samples: {len(TRAIN_DATASET)}")
log_string(f"Val samples: {len(TEST_DATASET)}")
log_string(f"Train batches: {len(TRAIN_DATALOADER)}")
log_string(f"Val batches: {len(TEST_DATALOADER)}")
log_string(f"Batch size: {FLAGS.batch_size}")
log_string(f"Num workers: {NUM_WORKERS}")


#################################################   network   #################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Network(cfg)
net.to(device)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
CHECKPOINT_PATH = FLAGS.checkpoint_path
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))


if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)




#################################################   training functions   ###########################################


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * cfg.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    current_lr = optimizer.param_groups[0]['lr']
    log_string(f"[TRAIN] Epoch {EPOCH_CNT:03d} started | lr={current_lr:.8f}")

    net.train()
    iou_calc = IoUCalculator(cfg)

    epoch_start_time = time.time()
    batch_start_time = time.time()

    for batch_idx, batch_data in enumerate(TRAIN_DATALOADER):
        for key in batch_data:
            if type(batch_data[key]) is list:
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
            if 'loss' in key or 'acc' in key or 'iou' in key:
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
    log_string('[TRAIN] mean IoU:{:.1f}'.format(mean_iou * 100))
    s = '[TRAIN] IoU: '
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)


def evaluate_one_epoch():
    stat_dict = {}
    net.eval()
    iou_calc = IoUCalculator(cfg)

    log_string(f"[VAL] Epoch {EPOCH_CNT:03d} started")
    val_start_time = time.time()

    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if type(batch_data[key]) is list:
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
            if 'loss' in key or 'acc' in key or 'iou' in key:
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
        log_string(f'[VAL] mean {key}: {stat_dict[key] / float(batch_idx + 1):.6f}')

    mean_iou, iou_list = iou_calc.compute_iou()
    val_time = time.time() - val_start_time

    log_string(f"[VAL] Epoch {EPOCH_CNT:03d} finished in {val_time / 60:.2f} min")
    log_string('[VAL] mean IoU:{:.1f}'.format(mean_iou * 100))
    s = '[VAL] IoU: '
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)


def train(start_epoch):
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
        log_string(f'**** EPOCH {epoch:03d} / {FLAGS.max_epoch - 1:03d} ****')
        log_string(str(datetime.now()))

        np.random.seed()
        train_one_epoch()

        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9:
            log_string(f'**** EVAL EPOCH {epoch:03d} START ****')
            evaluate_one_epoch()
            log_string(f'**** EVAL EPOCH {epoch:03d} END ****')

        save_dict = {
            'epoch': epoch + 1,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        try:
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()

        ckpt_path = os.path.join(LOG_DIR, 'checkpoint.tar')
        torch.save(save_dict, ckpt_path)
        log_string(f"[CKPT] Saved checkpoint to {ckpt_path} (next epoch: {epoch + 1})")


if __name__ == '__main__':

    train(start_epoch)

