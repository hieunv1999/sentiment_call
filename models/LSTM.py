# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()
        """
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights,
                                                   requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        # input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        # print(input.size())
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])
        return final_output
