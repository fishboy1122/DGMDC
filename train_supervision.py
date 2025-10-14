import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.backends
import torch.backends.cudnn
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
import torch, gc
from torch.nn.parallel import DataParallel

 
 
gc.collect()
torch.cuda.empty_cache()


# torch.multiprocessing.set_start_method('spawn')

torch.multiprocessing.set_sharing_strategy('file_system')

# gpu设置******************************
os.environ['CUDA_LAUNCH_BLOCKING']='1'

os.environ["CUDA_VISIBLE_DEVICES"]='0'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
# print("39-",torch.cuda.current_device())


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", default='config/vaihingen/DGMDC_config.py')

    return parser.parse_args()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.net=self.net
        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        # print("seg_pre",seg_pre.shape)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'].to(device), batch['gt_semantic_seg'].to(device)
        # img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.net(img)
        loss = self.loss(prediction, mask)

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # 打印显存使用情况
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        return {"loss": loss}

    def on_train_epoch_end(self):
        def custom_r2(accuracy, f1):  
            # 假设准确率和F1分数越高，模型的性能越好  
            # 这里只是一个简单的示例，你可以根据实际需求调整权重和计算方式  
            r2 = 0.5 * (accuracy ** 2 + f1 ** 2)  
            return r2  
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        R2 = custom_r2(OA,F1)
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA,
                      'R2': R2
                      }
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA,'train_R2':R2}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'].to(device), batch['gt_semantic_seg'].to(device)
        # img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        # 报错**************************
        # RuntimeError: CUDA error: device-side assert triggered
        # print("prediction",prediction.shape)
        # print("mask",mask.shape)
        # print("pre_mask",pre_mask.shape)
        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}
    
    def recall(y_true, y_pred):
        TP = torch.sum(y_true * torch.round(y_pred))
        recall = TP / (torch.sum(y_true) + 1e-8)
        return recall

    def precision(y_true, y_pred):
        TP = torch.sum(y_true * torch.round(y_pred))
        precision = TP / (torch.sum(torch.round(y_pred)) + 1e-8)
        return precision

    def fmeasure(y_true, y_pred):
        TP = torch.sum(y_true * torch.round(y_pred))
        precision = TP / (torch.sum(torch.round(y_pred)) + 1e-8)
        recall = TP / (torch.sum(y_true) + 1e-8)
        F1score = 2 * precision * recall / (precision + recall + 1e-8)
        return F1score

    def kappa_metrics(y_true, y_pred):
        TP = torch.sum(y_true * torch.round(y_pred))
        FP = torch.sum((1 - y_true) * torch.round(y_pred))
        FN = torch.sum(y_true * (1 - torch.round(y_pred)))
        TN = torch.sum((1 - y_true) * (1 - torch.round(y_pred)))
        totalnum = TP + FP + FN + TN
        p0 = (TP + TN) / (totalnum + 1e-8)
        pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (totalnum * totalnum + 1e-8)
        kappa_coef = (p0 - pe) / (1 - pe + 1e-8)
        return kappa_coef

    def OA(y_true, y_pred):
        TP = torch.sum(y_true * torch.round(y_pred))
        FP = torch.sum((1 - y_true) * torch.round(y_pred))
        FN = torch.sum(y_true * (1 - torch.round(y_pred)))
        TN = torch.sum((1 - y_true) * (1 - torch.round(y_pred)))
        totalnum = TP + FP + FN + TN
        overallAC = (TP + TN) / (totalnum + 1e-8)
        return overallAC
    
   
    
    def on_validation_epoch_end(self):
        
        def custom_r2(accuracy, f1):  
            # 假设准确率和F1分数越高，模型的性能越好  
            # 这里只是一个简单的示例，你可以根据实际需求调整权重和计算方式  
            r2 = 0.5 * (accuracy ** 2 + f1 ** 2)  
            return r2  
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()
        R2 = custom_r2(OA,F1)
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA,
                      'R2':R2}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA,'R2':R2}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)


    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback],strategy='auto',
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
    # trainer.fit(model=model)


if __name__ == "__main__":
   main()
