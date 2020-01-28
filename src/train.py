import torch
from torch import nn
import pandas as pd
import numpy as np


def train(parque_file_path,
          train_csv,
          num_classes,
          model_name,
          model_out_path,
          device,
          epochs=10):
    dataframe = pd.read_csv(train_csv)
    parque_file = pd.concat([pd.read_parquet(parque_file) for parque_file in parque_file_path])
    merged_dataframe = dataframe.merge(parque_file, left_on='image_id', right_on='image_id')
    image_data_list = [(img, grapheme_root, vowel_diacritic, consonant_diacritic)
                       for grapheme_root, vowel_diacritic, consonant_diacritic, img in
                       zip(merged_dataframe['grapheme_root'],
                            merged_dataframe['vowel_diacritic'],
                            merged_dataframe['consonant_diacritic'],
                            merged_dataframe.iloc[:,5:].values.reshape(-1, 137, 236))]
    print(len(image_data_list))


# def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None): grapheme_root    vowel_diacritic consonant_diacritic
train(["../data/bengaliai-cv19/train_image_data_{}.parquet".format(i) for i in range(4)],
      "../data/bengaliai-cv19/train.csv",10,"dads", 'fdwa', 'fs')