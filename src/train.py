import torch
from torch import nn
import pandas as pd
import numpy as np
import torchvision
from tqdm import tqdm
from torch.autograd import Variable
from loader import ImageLoader
from model import initialize_model
from torch.utils.tensorboard import SummaryWriter


def train(parque_file_path,
          train_csv,
          model_name,
          model_out_path,
          device,
          batch_size=128,
          epochs=10):
    writer = SummaryWriter('../runs')
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7
    loader = torch.utils.data.DataLoader(ImageLoader(train_csv, parque_file_path), batch_size=batch_size, shuffle=True, num_workers=8)
    model = initialize_model(model_name, n_grapheme+n_vowel+n_consonant, True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion =  nn.CrossEntropyLoss()
    global_step = 0
    model.train()
    for epoch in range(epochs):
        loss=0
        for img, labels in tqdm(loader):
            img = img.type(torch.FloatTensor).permute(0, 2, 3, 1).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            pred = model(img)
            preds = torch.split(pred, [n_grapheme, n_vowel, n_consonant], dim=1)
            loss_grapheme = criterion(preds[0], labels[:,0])
            loss_vowel = criterion(preds[1], labels[:,1])
            loss_consonant = criterion(preds[2], labels[:,2])
            total_loss = loss_grapheme+loss_vowel+loss_consonant
            total_loss.backward()
            optimizer.step()
            global_step += 1
            loss = loss+total_loss.item()
            grid = torchvision.utils.make_grid(img)
            writer.add_image('images', grid, 0)
            writer.add_graph(model, img)
            writer.add_scalar('Loss/train', total_loss.item(), global_step)
            writer.close()
        print(loss/len(loader))



# def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None): grapheme_root    vowel_diacritic consonant_diacritic
train(["../data/bengaliai-cv19/train_image_data_{}.parquet".format(i) for i in range(4)],
      "../data/bengaliai-cv19/train.csv","resnext50_32x4d", '../', 'cuda')