import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from torch.cuda.amp import autocast
from torch.cuda import amp
from sklearn import metrics
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
#import seaborn as sns
import matplotlib
import numpy as np
import umap
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torch import distributed as dist
def test_kf_2(model, criterion, data_loader, test_num, device):
    running_loss = 0.0
    CE_loss=0.0
    KL_loss=0.0

    correct_num = 0
    model.eval()
    batch_size = None
    with tqdm(enumerate(data_loader), total=len(data_loader)) as tbar:
        for index, data in tbar:
            x, y = data
            batch_size = x.shape[0]//2 if index == 0 else batch_size
            x = x.cuda().float().contiguous()
            y = y.cuda().long().contiguous()
            com1,com2,y_pred = model(x)
            _, pred = torch.max(y_pred, 1)
            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            loss1 = criterion[0](y_pred, y.long())
            loss2=criterion[1](com1,com2)
            loss=loss1+0.1*loss2
            running_loss += float(loss.item())
            CE_loss += float(loss1.item())
            KL_loss += float(0.1*loss2.item())

    batch_num = test_num // batch_size
    _loss = running_loss / (batch_num + 1)
    acc = correct_num / test_num * 100
    # f1 score
    f_y_true = y.cpu()
    f_y_pred = pred.cpu()
    f1 = metrics.f1_score(f_y_true, f_y_pred, average='macro')
    print(f'Test loss: {_loss:.4f}\tTest acc: {acc:.2f}%\tF1: {f1:.4f}')
    return acc, _loss, f1,CE_loss,KL_loss



def train_kf_fp16_2(model, criterion, optimizer, data_loader, train_num, epoch, epochs,
                  test_loader, test_num, enable_amp,
                  train_accL, train_lossL, test_accL, test_lossL, f1L, device, max_testacc, max_f1, best_test_loss,
                  early_stopping):
    #scaler = amp.GradScaler(enabled=enable_amp)
    model.train()
    runing_loss = 0.0
    CE_loss=0.0
    KL_loss=0.0
    correct_num = 0
    batch_size = None
    with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch{epoch + 1}/{epochs}', unit='it') as tbar:
        for index, (x, y) in tbar:
            #with amp.autocast(enabled=enable_amp):
            batch_size = x.shape[0]//2 if index == 0 else batch_size
            x = x.cuda().float().contiguous()   
            y = y.cuda().long().contiguous()
            com1,com2,y_pred = model(x)
            _, pred = torch.max(y_pred, 1)
            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            loss1 = criterion[0](y_pred, y)
            loss2=criterion[1](com1,com2)
            loss=loss1+0.1*loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            #optimizer.zero_grad()
            runing_loss += float(loss.item())
            CE_loss+=float(loss1.item())
            KL_loss+=float(0.1*loss2.item())


        batch_num = train_num // batch_size
        _loss = runing_loss / (batch_num + 1)

        acc = correct_num / train_num * 100

        print("start test...")
        test_acc, test_loss, f1, ce_loss,kl_loss= test_kf_2(model, criterion, test_loader, test_num, device)
        print(
            f'Epoch {epoch + 1}/{epochs}\tTrain loss: {_loss:.4f}\tTrain acc: {acc:.2f}%\tTest loss: {test_loss:.4f}\tTest acc: {test_acc:.2f}%\tf1score: {f1:.4f}\tce_loss:{ce_loss:.4f}\tkl_loss:{kl_loss:.4f}')

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print('best test_loss decreased to %.4f' % best_test_loss)
        early_stopping(test_loss, model)
        es = early_stopping.early_stop
        min_loss = early_stopping.val_loss_min
        if test_acc > max_testacc:
            max_testacc = test_acc
            max_f1 = f1

        train_accL.append(acc)
        train_lossL.append(_loss)
        test_accL.append(test_acc)
        test_lossL.append(test_loss)
        f1L.append(f1)

        return train_accL, train_lossL, test_accL, test_lossL, f1L, min_loss, es, max_testacc, max_f1, best_test_loss
        
