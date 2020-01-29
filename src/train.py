import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from loader import ImageLoader
from model import initialize_model


def train(parque_file_path,
          train_csv,
          num_classes,
          model_name,
          model_out_path,
          device,
          batch_size=32,
          epochs=10):
    n_grapheme = 168
    n_vowel = 11
    n_consonant = 7
    loader = torch.utils.data.DataLoader(ImageLoader(train_csv, parque_file_path), batch_size=batch_size, shuffle=True, num_workers=12)
    model = initialize_model(model_name, n_grapheme+n_vowel+n_consonant, True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
    criterion =  nn.CrossEntropyLoss()
    for epoch in range(epochs):
        loss=0
        for img, labels in tqdm(loader):
            img = img.type(torch.FloatTensor).permute(0, 2, 3, 1).to(device)
            y_pred = model(img)
            loss_grapheme = criterion(y_pred[:, :n_grapheme], labels[:,0])
            loss_vowel = criterion(y_pred[:, n_grapheme:n_grapheme+n_vowel], labels[:,1])
            loss_consonant = criterion(y_pred[:, n_grapheme+n_vowel:n_grapheme+n_vowel+n_consonant], labels[:,2])
            total_loss = loss_grapheme+loss_vowel+loss_consonant
            total_loss.backward()
            optimizer.step()
            loss = loss+total_loss.item()
        print(loss/len(loader))



# def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None): grapheme_root    vowel_diacritic consonant_diacritic
train(["../data/bengaliai-cv19/train_image_data_{}.parquet".format(i) for i in range(4)],
      "../data/bengaliai-cv19/train.csv",10,"resnet18", '../', 'cpu')