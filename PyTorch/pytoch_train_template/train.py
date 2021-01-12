# -*- coding: utf-8 -*-
# @File   : train.py
# @Author : zhkuo
# @Time   : 2020/12/17 9:38
# @Desc   : pytorch train template


# -*- coding: utf-8 -*-
# @File   : train_fashion_dataset.py
# @Author : zhkuo
# @Time   : 2020/12/21 9:40
# @Desc   :

import sys
import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import torch.utils.data

from utils.utils import *
from networks import get_model
from losses import loss_fn
# from data import trainset, testset
from data import CatDogDataset, get_transforms
from config import cfg, zlogger

# read the config
common_cfg = cfg['common']
path_cfg = cfg['path']
train_cfg = cfg['train']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed
seed_everything(common_cfg['seed'])

# dataset and dataloader
val_split = 0.1
data_path = r"datas/train"
img_paths = [os.path.join(data_path, tmp) for tmp in os.listdir(data_path)]

train_paths, val_paths = train_test_split(img_paths, test_size=val_split, random_state=common_cfg['seed'])

train_dataset = CatDogDataset(img_names=train_paths, transforms=get_transforms('train'))
val_dataset = CatDogDataset(img_names=val_paths, transforms=get_transforms('val'))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_cfg['train_bs'],
    shuffle=True,
    num_workers=train_cfg['num_workers']
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=train_cfg['val_bs'],
    shuffle=False,
    num_workers=train_cfg['num_workers']
)


# train loop
def train_val_one_epoch(_epoch, _total_epoches, _model, _data_loader, _loss, _optimizer, _device,
                        _scheduler=None, _scaler=None, _phase='train'):
    assert ('phase' not in ["train", "val"]), "phase can only be in train, val"
    acc = 0.
    if _phase == "train":
        _model.train()
    else:
        _model.eval()

    sample_num = 0
    loss_sum = 0
    image_preds_all = []
    image_targets_all = []
    pbar = tqdm(enumerate(_data_loader), total=len(_data_loader))
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(_device).float()
        labels = labels.to(_device).long()
        if _phase == "train":
            with autocast():
                # forward
                image_preds = _model(imgs)
                # calc loss
                loss = _loss(image_preds, labels)
                # backward
                _optimizer.zero_grad()
                # loss.backward()
                _scaler.scale(loss).backward()
                if ((step + 1) % train_cfg['accum_iter'] == 0) or ((step + 1) == len(_data_loader)):
                    # _optimizer.step()
                    _scaler.step(_optimizer)
                    _scaler.update()
                    if _scheduler is not None:
                        _scheduler.step()
        else:
            image_preds = _model(imgs)
            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [labels.detach().cpu().numpy()]

            loss = _loss(image_preds, labels)
        loss_sum += loss.item() * labels.shape[0]
        sample_num += labels.shape[0]
        if ((step + 1) % train_cfg['verbose_step'] == 0) or ((step + 1) == len(_data_loader)):
            description = f'Epoch {_epoch}/{_total_epoches} | {_phase}_loss = {loss_sum / sample_num:.4f}'
            pbar.set_description(description)
    if _phase == 'val':
        image_preds_all = np.concatenate(image_preds_all)
        image_targets_all = np.concatenate(image_targets_all)
        acc = (image_preds_all == image_targets_all).mean()
        zlogger('validation accuracy = {:.4f}'.format(acc))

    return acc


# instantiate network (which has been imported from *networks.py*)
n_class = 10

# model = MyNet('tf_efficientnet_b1_ns', pretrained=True, n_class=n_class)

# two way to freeze the parameters of conv:
#     1. set requires_grad=False
#     2. set conv lr=0 or very small
set_freeze = False
model = get_model('resnet50', pretrained=True, n_class=n_class, set_freeze=set_freeze)
model = model.to(device)

checkpoint_best_path = os.path.join(path_cfg['weight_path'], 'checkpoint_best.pth')


scaler = GradScaler()
# loss

myloss = loss_fn.to(device)

# if set_freeze=Fasle, can set conv layer lr=0 or very small
if set_freeze:
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
else:
    # set different learning rates for different layers
    fc_params_id = list(map(id, model.fc.parameters()))  # the addr of parameters
    base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': train_cfg['lr'] * 0},    # or lr*0.01 etc
        {'params': model.fc.parameters(), 'lr': train_cfg['lr']}], weight_decay=train_cfg['weight_decay'])


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg['T_0'], T_mult=1,
                                                                 eta_min=train_cfg['min_lr'], last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1,
#                                                 div_factor=25, max_lr=train_cfg['lr'],
#                                                 epochs=train_cfg['epochs'], steps_per_epoch=len(train_loader))
start_epoch = 0
# load checkpoint if needed/ wanted

resume = train_cfg['resume']
if resume:
    assert os.path.isfile(checkpoint_best_path)
    checkpoint = torch.load(checkpoint_best_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    zlogger('Load checkpoint at epoch {}.'.format(start_epoch))
    zlogger('Best accuracy so far {}.'.format(best_acc))
    zlogger('Last checkpoint restored')

# train loop
best_acc = 0.
current_acc = 0.
train_globel_step = 0
val_globel_step = 0
for epoch in range(start_epoch, train_cfg['epochs']):
    _ = train_val_one_epoch(epoch, train_cfg['epochs'], model, train_loader, myloss, _optimizer=optimizer,
                            _device=device, _scheduler=scheduler, _scaler=scaler, _phase='train')

    with torch.no_grad():
        current_acc = train_val_one_epoch(epoch, train_cfg['epochs'], model, val_loader, myloss, _optimizer=None,
                                          _device=device, _scheduler=None, _scaler=None, _phase='val')

    # save checkpoint
    is_best = current_acc > best_acc
    best_acc = max(current_acc, best_acc)
    checkpoint = {
        'best_acc': best_acc,
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(path_cfg['weight_path'], f'checkpoint_epoch_{epoch}.pth')
    best_model_path = os.path.join(path_cfg['weight_path'], 'best_weights.pth')
    torch.save(checkpoint, checkpoint_best_path)
    if is_best:
        torch.save(checkpoint, checkpoint_path)
        torch.save(model.state_dict(), best_model_path)

del model, optimizer, train_loader, val_loader, scaler, scheduler
torch.cuda.empty_cache()

