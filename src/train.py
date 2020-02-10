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

n_grapheme = 168
n_vowel = 11
n_consonant = 7
EVAL_STEP = 300
CUDA_LAUNCH_BLOCKING=1

def train(parque_file_path,
          train_csv,
          model_name,
          model_out_path,
          device,
          batch_size=32,
          epochs=10):
    writer = SummaryWriter('../runs')#'/media/mukesh/36AD331451677000/bengali_ai/runs/{}'.format(int(time.time())))
    labels_df = pd.read_csv(train_csv)[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
    image_df = pd.concat([pd.read_parquet(parque_file) for parque_file in parque_file_path])
    train_labels, val_labels, train_images, val_images = train_test_split(labels_df, image_df, test_size=0.1, random_state=11,stratify=labels_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']])
    print("train", train_labels.shape, "Val", val_labels.shape)
    train_loader = torch.utils.data.DataLoader(ImageLoader(train_labels, train_images), batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(ImageLoader(val_labels, val_images), batch_size=batch_size, shuffle=True, num_workers=8)
    model = BengaliModel(se_resnet('SEresnet18')).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion =  nn.CrossEntropyLoss()
    global_step = 0
    model.train()
    for epoch in range(epochs):
        loss = 0
        for img, labels in tqdm(train_loader):
            optimizer.zero_grad()
            img = img.type(torch.FloatTensor).permute(0, 2, 3, 1).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            grapheme_preds, vowel_preds, consonant_preds = model(img)
            loss_grapheme = nn.CrossEntropyLoss(weight=None)(grapheme_preds, labels[:,0])
            loss_vowel = nn.CrossEntropyLoss(weight=None)(vowel_preds, labels[:,1])
            loss_consonant = nn.CrossEntropyLoss(weight=None)(consonant_preds, labels[:,2])
            total_loss = loss_grapheme+loss_vowel+loss_consonant
            total_loss.backward(retain_graph=True)
            optimizer.step()
            global_step += 1
            loss = loss+total_loss.item()
            acc_grapheme = (grapheme_preds.argmax(1)==labels[:,0]).float().mean()
            acc_vowel = (vowel_preds.argmax(1)==labels[:,1]).float().mean()
            acc_consonant = (consonant_preds.argmax(1)==labels[:,2]).float().mean()
            acc = (acc_grapheme+acc_vowel+acc_consonant)/3
            if global_step%EVAL_STEP ==0:
                step_count = global_step/EVAL_STEP
                writer.add_scalar('Loss/train', total_loss.item(), step_count)
                writer.add_scalar('Loss/train/grapheme', loss_grapheme.item(), step_count)
                writer.add_scalar('Loss/train/vowel', loss_vowel.item(), step_count)
                writer.add_scalar('Loss/train/consonant', loss_consonant.item(), step_count)
                writer.add_scalar('Acc/train', acc, step_count)
                writer.add_scalar('Acc/train/grapheme', acc_grapheme, step_count)
                writer.add_scalar('Acc/train/vowel', acc_vowel, step_count)
                writer.add_scalar('Acc/train/consonant', acc_consonant, step_count)
                vl_grapheme, vl_vowel, vl_consonant, va_grapheme, va_vowel, va_consonant = evaluate(val_loader, model, nn.CrossEntropyLoss(), device)
                writer.add_scalar('Loss/val', vl_grapheme+vl_vowel+vl_consonant, step_count)
                writer.add_scalar('Loss/val/grapheme', vl_grapheme, step_count)
                writer.add_scalar('Loss/val/vowel', vl_vowel, step_count)
                writer.add_scalar('Loss/val/consonant', vl_consonant, step_count)
                writer.add_scalar('Acc/val', (va_grapheme+va_vowel+va_consonant)/3, step_count)
                writer.add_scalar('Acc/val/grapheme', va_grapheme, step_count)
                writer.add_scalar('Acc/val/vowel', va_vowel, step_count)
                writer.add_scalar('Acc/val/consonant', va_consonant, step_count)
                writer.close()
        print(loss/len(train_loader))

def evaluate(loader, model, criterion, device):
    loss_grapheme = []
    loss_vowel = []
    loss_consonant = []
    acc_grapheme = 0
    acc_vowel = 0
    acc_consonant = 0
    for img, labels in loader:
        img = img.type(torch.FloatTensor).permute(0, 2, 3, 1).to(device)
        labels = labels.type(torch.LongTensor).to(device)
        grapheme_preds, vowel_preds, consonant_preds = model(img)
        loss_grapheme.append(criterion(grapheme_preds, labels[:,0]).item())
        loss_vowel.append(criterion(vowel_preds, labels[:,1]).item())
        loss_consonant.append(criterion(consonant_preds, labels[:,2]).item())
        acc_grapheme+=(grapheme_preds.argmax(1)==labels[:,0]).float().mean()
        acc_vowel+=(vowel_preds.argmax(1)==labels[:,1]).float().mean()
        acc_consonant+=(consonant_preds.argmax(1)==labels[:,2]).float().mean()
    return np.mean(loss_grapheme), np.mean(loss_vowel), np.mean(loss_consonant), acc_grapheme/len(loader), acc_vowel/len(loader), acc_consonant/len(loader)


# def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None): grapheme_root    vowel_diacritic consonant_diacritic
train(["../data/bengaliai-cv19/train_image_data_{}.parquet".format(i) for i in range(4)],
      "../data/bengaliai-cv19/train.csv","resnet18", '../','cpu')