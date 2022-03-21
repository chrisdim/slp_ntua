import torch
from torch import nn
from attention import SelfAttention
import numpy as np

class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        num_embeddings, embed_dim = embeddings.shape #100 dimension
        self.embedding_layer = nn.Embedding(num_embeddings, embed_dim) # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # 4 - define a non-linear transformation of the representations
        self.linear = nn.Linear(embed_dim, 50)
        self.relu = nn.ReLU()  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.output = nn.Linear(50, output_size)  # EX5

    def forward(self, x, lengths, tfidf):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding_layer(x)  # EX6

        # Tf-Idf weights on the word embeddings
        shape = embeddings.shape
        for i in range(shape[0]):
            denominator = 0
            for j in range(shape[1]):
                embeddings[i][j] = tfidf[i][j]*embeddings[i][j] 
                denominator += tfidf[i][j]
            embeddings[i][:] = embeddings[i][:]/denominator
        
        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.sum(embeddings, dim=1)
        for i in range(lengths.shape[0]):
            representations[i] = representations[i] / lengths[i]  # EX6

        # 3 - transform the representations to new ones.
        representations = self.relu(self.linear(representations))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output(representations)  # EX6

        return logits

class LSTMTagger(nn.Module):

    def __init__(self, output_size, embeddings, hidden_dim, trainable_emb=False):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # 1 - define the embedding layer
        num_embeddings, embed_dim = embeddings.shape #100 dimension
        self.embedding_layer = nn.Embedding(num_embeddings, embed_dim) # EX4
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # EX4
        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim, output_size)


    def forward(self, x, lengths):      
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)

        # Sentence representation as the final hidden state of the model
        representations = torch.zeros(batch_size, self.hidden_dim).float()
        for i in range(lengths.shape[ 0]):
            # Discard padded zeros at the end
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]
        
        logits = self.linear(representations)
        
        return logits
    
class Concat_LSTM(nn.Module):

    def __init__(self, output_size, embeddings, hidden_dim, trainable_emb=False):
        super(Concat_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # 1 - define the embedding layer
        num_embeddings, embed_dim = embeddings.shape #100 dimension
        self.embedding_layer = nn.Embedding(num_embeddings, embed_dim) # EX4
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # EX4
        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(3*hidden_dim, output_size)


    def forward(self, x, lengths):      
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)

        # Sentence representation as the final hidden state of the model
        representations = torch.zeros(batch_size, self.hidden_dim).float()
        mean_pooling = torch.sum(ht, dim=1)
        for i in range(lengths.shape[ 0]):
            # Discard padded zeros at the end
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]

            # Mean pooling of LSTM outputs
            mean_pooling[i] = mean_pooling[i] / lengths[i]

        
        # Max pooling of LSTM outputs
        h = torch.transpose(ht, 1, 2)  # [N,L,C] -> [N,C,L]
        m = nn.MaxPool1d(max_length)
        max_pooling = m(h)
        max_pooling=max_pooling.squeeze() # discard dimension that is 1
        
        # Concatenate all three of them
        representations = torch.cat((representations, mean_pooling, max_pooling),1)
        
        logits = self.linear(representations)
        
        return logits
    
class LSTM_Attention(nn.Module):

    def __init__(self, output_size, embeddings, hidden_dim, attention_dim, non_linearity, focus ='hidden_state', trainable_emb=False):
        super(LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.focus = focus
        
        # 1 - define the embedding layer
        num_embeddings, embed_dim = embeddings.shape #100 dimension
        self.embedding_layer = nn.Embedding(num_embeddings, embed_dim) # EX4
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # EX4
        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        
        self.attention = SelfAttention(attention_dim, non_linearity)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(attention_dim, output_size)


    def forward(self, x, lengths):      
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)

        # apply attention to get Sentence representation
        if self.focus == 'embeddings':
            #3.1 Attention on Embeddings
            representations, att_scores = self.attention(embeddings, lengths)
        else:
            #3.2 Attention on Hidden States of LSTM
            representations, att_scores = self.attention(ht, lengths)
        
        logits = self.linear(representations)
        
        return logits
    
class Bidirectional_Concat_LSTM(nn.Module):

    def __init__(self, output_size, embeddings, hidden_dim, trainable_emb=False):
        super(Bidirectional_Concat_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # 1 - define the embedding layer
        num_embeddings, embed_dim = embeddings.shape #100 dimension
        self.embedding_layer = nn.Embedding(num_embeddings, embed_dim) # EX4
        
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # EX4
        
        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, 
                            bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(hidden_dim*6, output_size)


    def forward(self, x, lengths):      
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)

        
        # Sentence representation as the final hidden state of the model
        first_half = torch.zeros(batch_size, self.hidden_dim).float()
        second_half = torch.zeros(batch_size, self.hidden_dim).float()
        
        # Mean pooling of LSTM outputs
        mean_pooling1 = torch.sum(ht[:,:,:self.hidden_dim], dim=1)
        mean_pooling2 = torch.sum(ht[:,:,self.hidden_dim:], dim=1)
        
        for i in range(lengths.shape[0]):
            # Discard padded zeros at the end
            last1 = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            last2 = lengths[i] - 1 if lengths[-(i+1)] <= max_length else max_length - 1
            first_half[i] = ht[i, last1, :self.hidden_dim]
            second_half[i] = ht[-(i+1), last2, self.hidden_dim:]
            
            mean_pooling1[i] = mean_pooling1[i] / lengths[i]
            mean_pooling2[i] = mean_pooling2[i] / lengths[-(i+1)]
        

        # Max pooling of LSTM outputs
        h = torch.transpose(ht[:,:,:self.hidden_dim], 1, 2)  # [N,L,C] -> [N,C,L]
        m = nn.MaxPool1d(max_length)
        max_pooling1 = m(h)
        max_pooling1=max_pooling1.squeeze() # discard dimension that is 1
        
        representations1 = torch.cat((first_half, mean_pooling1, max_pooling1),1)

        h = torch.transpose(ht[:,:,self.hidden_dim:], 1, 2)  # [N,L,C] -> [N,C,L]
        max_pooling2 = m(h)
        max_pooling2=max_pooling2.squeeze() # discard dimension that is 1
        
        representations2 = torch.cat((second_half, mean_pooling2, max_pooling2),1)
      
        ### COMBINE BOTH DIRECTIONS ######################################
        representations = torch.cat((representations1, representations2),1)
        
        logits = self.linear(representations)
        
        return logits
    
class Bidirectional_LSTM_Attention(nn.Module):

    def __init__(self, output_size, embeddings, hidden_dim, attention_dim, non_linearity, focus ='hidden_state', trainable_emb=False):
        super(Bidirectional_LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.focus = focus
        
        # 1 - define the embedding layer
        num_embeddings, embed_dim = embeddings.shape #100 dimension
        self.embedding_layer = nn.Embedding(num_embeddings, embed_dim) # EX4
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embeddings))  # EX4
        # 3 - define if the embedding layer will be frozen or finetuned
        self.embedding_layer.weight.requires_grad = trainable_emb  # EX4

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True)
        
        self.attention = SelfAttention(attention_dim, non_linearity)

        # The linear layer that maps from hidden state space to tag space
        self.linear = nn.Linear(attention_dim, output_size)


    def forward(self, x, lengths, tfidf):      
        batch_size, max_length = x.shape
        embeddings = self.embedding_layer(x)
        
        # Tf-Idf weights on the word embeddings
        shape = embeddings.shape
        for i in range(shape[0]):
            denominator = 0
            for j in range(shape[1]):
                embeddings[i][j] = tfidf[i][j]*embeddings[i][j] 
                denominator += tfidf[i][j]
            embeddings[i][:] = embeddings[i][:]/denominator
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first =True, enforce_sorted =False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first =True)

        # apply attention to get Sentence representation
        if self.focus == 'embeddings':
            #3.1 Attention on Embeddings
            representations, att_scores = self.attention(embeddings, lengths)
        else:
            #3.2 Attention on Hidden States of LSTM
            representations, att_scores = self.attention(ht, lengths)
  
        logits = self.linear(representations)
        
        return logits #, att_scores