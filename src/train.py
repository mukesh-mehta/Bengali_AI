import torch
import time
from torch import nn
from tqdm import tqdm
from loader import ImageLoader
from model import initialize_model
from torch.utils.tensorboard import SummaryWriter


def train(parque_file_path,
          train_csv,
          model_name,
          model_out_path,
          device,
          batch_size=256,
          epochs=10):
    writer = SummaryWriter('/media/mukesh/36AD331451677000/bengali_ai/runs/{}'.format(int(time.time())))
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
        loss = 0
        for img, labels in tqdm(loader):
            optimizer.zero_grad()
            img = img.type(torch.FloatTensor).permute(0, 2, 3, 1).to(device)
            labels = labels.type(torch.LongTensor).to(device)
            pred = model(img)
            preds = torch.split(pred, [n_grapheme, n_vowel, n_consonant], dim=1)
            loss_grapheme = criterion(preds[0], labels[:,0])
            loss_vowel = criterion(preds[1], labels[:,1])
            loss_consonant = criterion(preds[2], labels[:,2])
            total_loss = loss_grapheme+loss_vowel+loss_consonant
            total_loss.backward(retain_graph=True)
            optimizer.step()
            global_step += 1
            loss = loss+total_loss.item()
            acc_grapheme = (preds[0].argmax(1)==labels[:,0]).float().mean()
            acc_vowel = (preds[1].argmax(1)==labels[:,1]).float().mean()
            acc_consonant = (preds[2].argmax(1)==labels[:,2]).float().mean()
            acc = (acc_grapheme+acc_vowel+acc_consonant)/3
            writer.add_scalar('Loss/train', total_loss.item(), global_step)
            writer.add_scalar('Loss/train/grapheme', loss_grapheme.item(), global_step)
            writer.add_scalar('Loss/train/vowel', loss_vowel.item(), global_step)
            writer.add_scalar('Loss/train/consonant', loss_consonant.item(), global_step)
            writer.add_scalar('Acc/train', acc, global_step)
            writer.add_scalar('Acc/train/grapheme', acc_grapheme, global_step)
            writer.add_scalar('Acc/train/vowel', acc_vowel, global_step)
            writer.add_scalar('Acc/train/consonant', acc_consonant, global_step)
            writer.close()
        print(loss/len(loader))



# def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None): grapheme_root    vowel_diacritic consonant_diacritic
train(["../data/bengaliai-cv19/train_image_data_{}.parquet".format(i) for i in range(4)],
      "../data/bengaliai-cv19/train.csv","resnext50_32x4d", '../', 'cuda')