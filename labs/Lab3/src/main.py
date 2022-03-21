import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTMTagger, Concat_LSTM, LSTM_Attention, Bidirectional_Concat_LSTM, Bidirectional_LSTM_Attention
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.100d.txt")
#EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.100d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 100

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 20
DATASET = "MR"  # options: "MR", "Semeval2017A"
#DATASET="Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # EX1
y_test = le.fit_transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size


"""
print("First 10 Labels of Train Set:")
for i in range(10):
    print(y_train[i])

print("Labels - Numbers Matching:")
for i in range(le.classes_.size):
    print(i," - ", le.classes_[i], "\n")
"""


# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)


"""
for i in range(10):
    print(train_set.data[i])
    
for i in [5 , 85, 96, 105, 582]:
    print(train_set.data[i])
    (embedding,label,length) = train_set[i]
    print("\nCode Form: ", embedding)
    print("Label: ", label)
    print("Sentence Original Length: ", length,"\n")
"""

# EX4 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)  # EX7
            

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
"""

# Prelab: Baselinne DNN Model
model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)


# Part 2.1: LSTM Model
model = LSTMTagger(output_size=n_classes,
                    embeddings=embeddings, hidden_dim = 50,
                    trainable_emb=EMB_TRAINABLE)

# Part 2.2: LSTM Model
model = Concat_LSTM(output_size=n_classes,
                    embeddings=embeddings, hidden_dim = 100,
                    trainable_emb=EMB_TRAINABLE)



# Part 3.1: LSTM Model with Attention on embeddings
model = LSTM_Attention(output_size=n_classes,
                    embeddings=embeddings, hidden_dim = 50,
                    attention_dim = 100, non_linearity="tanh",
                    focus = 'embeddings',
                    trainable_emb=EMB_TRAINABLE)

# Part 3.2: LSTM Model with Attention on Hidden States
model = LSTM_Attention(output_size=n_classes,
                    embeddings=embeddings, hidden_dim = 50,
                    attention_dim = 50, non_linearity="tanh", 
                    focus = 'hidden_state',
                    trainable_emb=EMB_TRAINABLE)


# Part 4.1: Bidirectional LSTM Model
model = Bidirectional_Concat_LSTM(output_size=n_classes,
                    embeddings=embeddings, hidden_dim = 100,
                    trainable_emb=EMB_TRAINABLE)

"""
# Part 4.2: Bidirectional LSTM Model with Attention on Hidden States
model = Bidirectional_LSTM_Attention(output_size=n_classes,
                    embeddings=embeddings, hidden_dim = 50,
                    attention_dim = 100, non_linearity="tanh", 
                    focus = 'hidden_states',
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
if n_classes ==2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss() # EX8
parameters = model.parameters()  # EX8
optimizer = torch.optim.Adam(parameters, lr=0.001)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
loss_test = []
loss_train = []
PATH = "./model"


for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, n_classes)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold), _ = eval_dataset(train_loader,
                                                            model,
                                                            criterion, n_classes)

    test_loss, (y_test_pred, y_test_gold), _ = eval_dataset(test_loader,
                                                         model,
                                                         criterion, n_classes)
    loss_test.append(test_loss)
    loss_train.append(train_loss)
    
    
    
    
    
    # Evaluate Train Dataset with Metrics
    print("===================EVALUATE MODEL====================")
    train_accuracy = 0
    train_recall = 0
    train_f1 = 0
    for i in range(len(y_train_gold)):
        train_accuracy += metrics.accuracy_score(y_train_gold[i], y_train_pred[i])
        train_recall += metrics.recall_score(y_train_gold[i], y_train_pred[i], average='macro')
        train_f1 += metrics.f1_score(y_train_gold[i], y_train_pred[i], average='macro')
    
    train_accuracy = train_accuracy/len(y_train_gold)
    train_recall = train_recall/len(y_train_gold)
    train_f1 = train_f1/len(y_train_gold)
    print("TRAIN DATASET EVALUATION:")
    print("Epoch Loss = ", train_loss)
    print("Accuracy = ", train_accuracy)
    print("Recall = ", train_recall)    
    print("F1 Score = ", train_f1)
    
    # Evaluate Test Dataset with Metrics
    test_accuracy = 0
    test_recall = 0
    test_f1 = 0
    for i in range(len(y_test_gold)):
        test_accuracy += metrics.accuracy_score(y_test_gold[i], y_test_pred[i])
        test_recall += metrics.recall_score(y_test_gold[i], y_test_pred[i], average='macro')
        test_f1 += metrics.f1_score(y_test_gold[i], y_test_pred[i], average='macro')
    
    test_accuracy = test_accuracy/len(y_test_gold)
    test_recall = test_recall/len(y_test_gold)
    test_f1 = test_f1/len(y_test_gold)
    print("\nTEST DATASET EVALUATION:")
    print("Epoch Loss = ", test_loss)
    print("Accuracy = ", test_accuracy)
    print("Recall = ", test_recall)    
    print("F1 Score = ", test_f1) 

# Plot Train & Test Loss - Epochs
plt.figure(1)
x = np.arange(1,EPOCHS+1,1)
plt.stem(x, np.array(loss_train),label="MR Dataset")
plt.xlabel("Epochs")
plt.ylabel("Train Loss")
#plt.title("Training Set Loss of DNN for 50 Epochs")
plt.legend()
plt.show()

plt.figure(2)
plt.stem(x, np.array(loss_test),label="MR Dataset")
plt.xlabel("Epochs")
plt.ylabel("Test Loss")
#plt.title("Test Set Loss of DNN for 50 Epochs")
plt.legend()
plt.show()

# Save model for later use
torch.save(model, PATH)





