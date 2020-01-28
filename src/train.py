import torch
from torch import nn
import pandas as pd
import numpy as np


def train(image_file_path, train_csv, num_classes, model_name, model_out_path, device, epochs=10):
    dataframe = pd.read_csv(train_csv)



def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None):
