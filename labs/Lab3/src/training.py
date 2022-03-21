import math
import sys
from torch.nn.utils.rnn import pad_sequence
import torch


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer, n_classes):
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device
    torch.autograd.set_detect_anomaly(True)

    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths, tfidf = batch
        inputs = pad_sequence(inputs)
        tfidf = pad_sequence(tfidf)
        
        # Also convert target labels to logits in case of BCEWithLogitsLoss
        if n_classes==2:
            logit_labels=[]
            for item in labels:
                if item==0:
                    logit_labels.append([1.0,0.0])
                else:
                    logit_labels.append([0.0, 1.0])
            labels = torch.tensor(logit_labels)
            
        
        # move the batch tensors to the right device
        inputs, labels, lengths, tfidf = inputs.to(device), labels.to(device), lengths.to(device), tfidf.to(device)  # EX9

        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        model.zero_grad()  # EX9

        # Step 2 - forward pass: y' = model(x)
        outputs = model(inputs, lengths, tfidf)  # EX9

        # Step 3 - compute loss: L = loss_function(y, y')
        loss = loss_function(outputs, labels)  # EX9

        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()  # EX9

        # Step 5 - update weights
        optimizer.step()  # EX9

        running_loss += loss.data.item()

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / len(dataloader)


def eval_dataset(dataloader, model, loss_function, n_classes):
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels
    attentions_scores = []
    
    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # get the inputs (batch)
            inputs, labels, lengths, tfidf = batch
            inputs = pad_sequence(inputs)
            tfidf = pad_sequence(tfidf)
            
            gold_labels = []
            gold_labels.append(labels.tolist())
            
            # Also convert target labels to logits in case of BCEWithLogitsLoss
            if n_classes==2:
                logit_labels=[]
                for item in labels:
                    if item == 0:
                        logit_labels.append([1.0, 0.0])
                    else:
                        logit_labels.append([0.0, 1.0])
                labels = torch.tensor(logit_labels)

            # Step 1 - move the batch tensors to the right device
            inputs, labels, lengths, tfidf = inputs.to(device), labels.to(device), lengths.to(device), tfidf.to(device)  # EX9


            # Step 2 - forward pass: y' = model(x)
            outputs = model(inputs, lengths, tfidf)  # EX9
            
            # if attention is used, also return att_scores
            #outputs, att_scores = model(inputs, lengths)
            #attentions_scores.append(att_scores)
            
            # Step 3 - compute loss.
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            loss = loss_function(outputs, labels)  # EX9

            # Step 4 - make predictions (class = argmax of posteriors)
            prediction = []
            for est in outputs:
                index = torch.argmax(est)
                prediction.append(index)  # EX9

            # Step 5 - collect the predictions, gold labels and batch loss
            y_pred.append(prediction)
            y.append(gold_labels[0])   # EX9

            running_loss += loss.data.item()

    return running_loss / len(dataloader), (y_pred, y) , attentions_scores
