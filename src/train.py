import torch
from torch import nn
import numpy as np


def train(train, val, test, model_name, num_classes, model_out_path, device, epochs = 10):
    print("Training on : {}, Validating on :{}".format(train, val))
    # Compute class weight
    train_df = pd.read_csv(config.TASK1["Folds"]+"/"+train, sep="\t")
    class_weights = torch.FloatTensor([compute_class_weight('balanced', [0,1], train_df['has_def'].values)[1]]).to(device)
    del train_df #delete df

    # Get iterators and Vocab_instance
    train_iter, val_iter, test_iter, TEXT = get_iterators(train, val, test, device, vectors=vectors)

    train_dl = BatchWrapper(train_iter, 'text', ['has_def'])
    valid_dl = BatchWrapper(val_iter, 'text', ['has_def'])
    test_dl = BatchWrapper(test_iter, 'text', ['has_def'])
    
    # model = SimpleLSTMBaseline(300, TEXT.vocab.vectors, emb_dim=300).to(device)
    model = DeepMoji(embedding_vector=TEXT.vocab.vectors,
                    vocab_size=len(TEXT.vocab),
                    embedding_dim=300,
                    hidden_state_size=256,
                    num_layers=2,
                    output_dim=1,
                    dropout=0.5,
                    bidirectional=True,
                    pad_idx=TEXT.vocab.stoi["<PAD>"]
                ).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    # loss_func = nn.BCEWithLogitsLoss()

    # print("Epoch, Training Loss, Training f-score, Validation Loss, Validation f-score")
    # Best score
    best_score = 0.0
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        model.train() # turn on training mode
        train_preds = []
        train_truth = []
        for x, y in tqdm(train_dl):
            opt.zero_grad()

            preds = model(x[0], x[1]) # x[0] is text sequence, x[1] is len of sequence
            # loss = loss_func(preds, y)
            loss = f1_loss(preds, y, weights=class_weights)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            opt.step()
            train_preds.extend(nn.Sigmoid()(preds).detach().cpu().numpy())
            train_truth.extend(y.cpu().numpy())
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_dl)
        
        # evaluate on validation set
        val_loss, val_preds, val_truth = evaluate(valid_dl, model, f1_loss, device, weights = class_weights) #change loss here

        train_preds = np.where(np.array(train_preds)<0.5, 0, 1).flatten()
        train_fscore = f1_score(train_truth, train_preds)
        val_fscore = f1_score(val_truth, val_preds)
        # print('{}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(epoch, epoch_loss, train_fscore, val_loss, val_fscore))
        print('Epoch {}, Training Loss {:.4f}, Training f-score {:.4f}, Validation Loss {:.4f}, Validation f-score {:.4f}'.format(epoch, epoch_loss, train_fscore, val_loss, val_fscore))
        print("classification report Train")
        print(classification_report(train_truth, train_preds))
        print("classification report Validation")
        print(classification_report(val_truth, val_preds))
        if val_fscore > best_score:
            best_score = val_fscore
            torch.save(model.state_dict(), model_out_path)
            print("Saving model with best_score {}".format(best_score))

    test_loss, test_preds, test_truth = evaluate(test_dl, model, f1_loss, device, checkpoint = model_out_path)#change loss here
    test_fscore = f1_score(test_truth, test_preds)
    print("Test Loss: {:.4f}, Test F1-score {:.4f}".format(test_loss, test_fscore))
    print("classification report Test")
    print(classification_report(test_truth, test_preds))
    return


def evaluate(loader, model, loss_func, device, checkpoint=None, weights=None):
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))
        model.to(device)
    val_loss = 0.0
    val_preds = []
    val_truth = []
    model.eval()  # turn on evaluation mode
    for x, y in loader:
        preds = model(x)
        loss = loss_func(preds, y, weights=weights)
        val_loss += loss.item()
        val_preds.extend(nn.Sigmoid()(preds).detach().cpu().numpy())
        val_truth.extend(y.cpu().numpy())

    val_loss /= len(loader)
    val_preds = np.where(np.array(val_preds) < 0.5, 0, 1).flatten()
    return val_loss, val_preds, val_truth
