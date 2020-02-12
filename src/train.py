import torch
import time
import pandas as pd
import numpy as np
from utils import log_writer
from torch import nn
from tqdm import tqdm
from loader import ImageLoader
from model import BengaliModel, se_resnet
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet


n_grapheme = 168
n_vowel = 11
n_consonant = 7
EVAL_STEP = 1000
CUDA_LAUNCH_BLOCKING=1

def train_model(parque_file_path, train_csv, model_name, model_out_path, device, logs_path, batch_size=16, epochs=10):
    writer = SummaryWriter(logs_path)
    labels_df = pd.read_csv(train_csv)[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    image_df = pd.concat([pd.read_parquet(parque_file) for parque_file in parque_file_path])
    train_labels, val_labels, train_images, val_images = train_test_split(labels_df, image_df, test_size=0.1, random_state=11,stratify=labels_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']])
    print("train", train_labels.shape, "Val", val_labels.shape)
    train_loader = torch.utils.data.DataLoader(ImageLoader(train_labels, train_images), batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(ImageLoader(val_labels, val_images), batch_size=batch_size, shuffle=True, num_workers=8)
    # model = BengaliModel(se_resnet(model_name, activation='relu')).to(device)
    model = BengaliModel(EfficientNet.from_name('efficientnet-b1')).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=5)
    criterion =  nn.CrossEntropyLoss()
    best_loss = 10.0
    for epoch in range(epochs):
        model, optimizer, loss_grapheme, loss_vowel, loss_consonant, acc_grapheme, acc_vowel, acc_consonant = train(train_loader, model, criterion, device, optimizer, scheduler=None)
        total_loss = loss_grapheme+loss_vowel+loss_consonant
        print("train loss:", total_loss)
        torch.cuda.empty_cache()
        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Loss/train/grapheme', loss_grapheme, epoch)
        writer.add_scalar('Loss/train/vowel', loss_vowel, epoch)
        writer.add_scalar('Loss/train/consonant', loss_consonant, epoch)
        writer.add_scalar('Acc/train', (acc_grapheme+acc_vowel+acc_consonant)/3)
        writer.add_scalar('Acc/train/grapheme', acc_grapheme, epoch)
        writer.add_scalar('Acc/train/vowel', acc_vowel, epoch)
        writer.add_scalar('Acc/train/consonant', acc_consonant, epoch)
        vl_grapheme, vl_vowel, vl_consonant, va_grapheme, va_vowel, va_consonant = evaluate(val_loader, model, nn.CrossEntropyLoss(), device)
        torch.cuda.empty_cache()
        val_loss = vl_grapheme + vl_vowel + vl_consonant
        writer.add_scalar('Loss/val', vl_grapheme+vl_vowel+vl_consonant, epoch)
        writer.add_scalar('Loss/val/grapheme', vl_grapheme, epoch)
        writer.add_scalar('Loss/val/vowel', vl_vowel, epoch)
        writer.add_scalar('Loss/val/consonant', vl_consonant, epoch)
        writer.add_scalar('Acc/val', (va_grapheme+va_vowel+va_consonant)/3, epoch)
        writer.add_scalar('Acc/val/grapheme', va_grapheme, epoch)
        writer.add_scalar('Acc/val/vowel', va_vowel, epoch)
        writer.add_scalar('Acc/val/consonant', va_consonant, epoch)
        writer.close()
        if val_loss < best_loss:
            torch.save(model.state_dict(), model_out_path)
            best_loss = val_loss
        

def train(loader, model, criterion, device, optimizer, scheduler=None):
    rloss_grapheme = []
    rloss_vowel = []
    rloss_consonant = []
    acc_grapheme = 0
    acc_vowel = 0
    acc_consonant = 0
    model.train()
    for img, labels in tqdm(loader):
        optimizer.zero_grad()
        img = img.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.LongTensor).to(device)
        grapheme_preds, vowel_preds, consonant_preds = model(img)
        loss_grapheme = criterion(grapheme_preds, labels[:,0])
        loss_vowel = criterion(vowel_preds, labels[:,1])
        loss_consonant = criterion(consonant_preds, labels[:,2])
        total_loss = loss_grapheme+loss_vowel+loss_consonant
        total_loss.backward(retain_graph=True)
        optimizer.step()
        if scheduler:
            scheduler.step()
        rloss_grapheme.append(loss_grapheme.item())
        rloss_vowel.append(loss_vowel.item())
        rloss_consonant.append(loss_consonant.item())
        acc_grapheme += (grapheme_preds.argmax(1) == labels[:, 0]).float().mean()
        acc_vowel += (vowel_preds.argmax(1) == labels[:, 1]).float().mean()
        acc_consonant += (consonant_preds.argmax(1) == labels[:, 2]).float().mean()
    return model, optimizer, np.mean(rloss_grapheme), np.mean(rloss_vowel), np.mean(rloss_consonant), acc_grapheme/len(loader), acc_vowel/len(loader), acc_consonant/len(loader)


def evaluate(loader, model, criterion, device):
    loss_grapheme = []
    loss_vowel = []
    loss_consonant = []
    acc_grapheme = 0
    acc_vowel = 0
    acc_consonant = 0
    model.eval()
    for img, labels in tqdm(loader):
        img = img.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.LongTensor).to(device)
        grapheme_preds, vowel_preds, consonant_preds = model(img)
        loss_grapheme.append(criterion(grapheme_preds, labels[:,0]).item())
        loss_vowel.append(criterion(vowel_preds, labels[:,1]).item())
        loss_consonant.append(criterion(consonant_preds, labels[:,2]).item())
        acc_grapheme+=(grapheme_preds.argmax(1)==labels[:,0]).float().mean()
        acc_vowel+=(vowel_preds.argmax(1)==labels[:,1]).float().mean()
        acc_consonant+=(consonant_preds.argmax(1)==labels[:,2]).float().mean()
    return np.mean(loss_grapheme), np.mean(loss_vowel), np.mean(loss_consonant), acc_grapheme/len(loader), acc_vowel/len(loader), acc_consonant/len(loader)


if __name__ == '__main__':
    train_model(["../data/bengaliai-cv19/train_image_data_{}.parquet".format(i) for i in range(4)],
      "../data/bengaliai-cv19/train.csv","SEresnet34", '/media/mukesh/36AD331451677000/bengali_ai/models/model.pt','cuda',
      '/media/mukesh/36AD331451677000/bengali_ai/runs/{}'.format(int(time.time())))