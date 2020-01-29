import torch
from torch import nn
import pandas as pd
import numpy as np
from loader import ImageLoader


def train(parque_file_path,
          train_csv,
          num_classes,
          model_name,
          model_out_path,
          device,
          batch_size=16,
          epochs=10):
    loader = torch.utils.data.DataLoader(ImageLoader(train_csv, parque_file_path), batch_size=batch_size, shuffle=True, num_workers=8)
    for img, labels in loader:
        print(labels)
        break



# def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None): grapheme_root    vowel_diacritic consonant_diacritic
train(["../data/bengaliai-cv19/train_image_data_{}.parquet".format(i) for i in range(4)],
      "../data/bengaliai-cv19/train.csv",10,"dads", 'fdwa', 'fs')