import sys
import torch
import pdb
import random
import string
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import time
import math
import copy


class Utils():
    @staticmethod
    def time_since(since):
        s = time.time() - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

class Inferencer():
    def __init__(self, model_file, voc_file, device):
        self.model_file = model_file
        self.device = device

        # load a trained model
        self.lstm = torch.load(self.model_file).to(self.device)
        #self.vocabularyLoader = VocabularyLoader("cre-utf8.txt", self.device)
        #self.vocabularyLoader = VocabularyLoader("docker.txt", self.device)
        #self.vocabularyLoader = VocabularyLoader("caiding.txt", self.device)
        #self.vocabularyLoader = VocabularyLoader("suanfa.txt", self.device)
        self.vocabularyLoader = VocabularyLoader(voc_file, self.device)


    def getText(self, prime_str, predict_len, temperature=0.8):
        prime_input = self.vocabularyLoader.char_tensor(prime_str)
        hidden = self.lstm.init_hidden().to(self.device)
        cell_state = self.lstm.init_cell_state().to(self.device)
        for p in range(len(prime_str) - 1):
            output, hidden, cell_state = self.lstm(prime_input[p], hidden, cell_state)

        inp = prime_input[-1]
        sentence = prime_str
        for i in range(predict_len):
            output, hidden, cell_state = self.lstm(inp, hidden, cell_state)
            output_dist = output.div(temperature).exp()

            #only use the first element in the array
            predict = torch.multinomial(output_dist, num_samples=1)[0]
            #predict = torch.multinomial(output_dist, num_samples=1)#it's wrong

            # 0 means the only one element in the string
            try:
                predict_char = self.vocabularyLoader.index2char[predict.item()]
            except Exception as e:
                pdb.set_trace()
                print(e)
            inp = predict
            sentence += predict_char

        return sentence


class Trainer():
    def __init__(self, voc_size, device):
        self.embedding_dim = 128
        self.hidden_size = 128
        self.learning_rate = 1e-3
        self.learning_rate_decay=20
        self.learning_rate_decay_count=1
        self.n_step = 500000
        self.voc_size = voc_size
        self.device = device


    def __create_model(self):
        lstm = LSTM(self.voc_size, self.embedding_dim, self.hidden_size)
        return lstm


    def train_within_step(self, lstm, optimizer, criterion, inp, target, chunk_len):
        hidden = lstm.init_hidden().to(self.device)
        cell_state = lstm.init_cell_state().to(self.device)
        lstm.zero_grad()
        loss = 0

        for c in range(chunk_len-1):
            try:
                output, hidden, cell_state = lstm(inp[c], hidden, cell_state)
                # the first parameter in criterion should be in the shape (minibatch, others)
                # the second parameter also should be in the shape (minibatch, others)
                loss += criterion(output.unsqueeze(0), target[c].unsqueeze(0))
                #pdb.set_trace()
            except Exception as e:
                pdb.set_trace()
                print(e)

        loss.backward()
        optimizer.step()

        # return loss.data.item() / (chunk_len-1)
        return loss.data.item()

    def train_model(self, dataloader):
        lstm = self.__create_model()
        lstm = lstm.to(self.device)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()

        start = time.time()
        for step in range(1, self.n_step+1):
            # if step == self.n_step / self.learning_rate_decay*self.learning_rate_decay_count:
            #     self.learning_rate /= 10
            #     optimizer = torch.optim.Adam(lstm.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            #     self.learning_rate_decay_count+=7
            #     #print (self.learning_rate)
            input, target = dataloader.next_chunk()
            loss = self.train_within_step(lstm, optimizer, criterion, input, target, dataloader.chunk_len)
            print('[%s (%d %d%%) %.4f]' % (Utils.time_since(start), step, step / self.n_step* 100, loss))


            f=open("procedure.txt","a+")
            f.write('[%s (%d %d%%) %.4f]' % (Utils.time_since(start), step, step / self.n_step* 100, loss)+'\n')
            f.close()

            if step % 1000 == 0:
                torch.save(lstm, "lstm_"+"{}".format(step)+".model")

class LSTM(nn.Module):
    def __init__(self, voc_size, embedding_dim, hidden_size):
        super(LSTM, self).__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(voc_size, embedding_dim)
        #forget gate
        self.wf = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.uf = nn.Linear(hidden_size, hidden_size, bias = True)
        #input gate
        self.wi = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.ui = nn.Linear(hidden_size, hidden_size, bias =True)
        #ouput gate
        self.wo = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.uo = nn.Linear(hidden_size, hidden_size, bias = True)
        #for updating cell state vector
        self.wc = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.uc = nn.Linear(hidden_size, hidden_size, bias = True)
        #gate's activation function
        self.sigmoid = nn.Sigmoid()
        #activation function on the updated cell state
        self.tanh = nn.Tanh()
        #distribution of the prediction
        self.out = nn.Linear(hidden_size, voc_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden, cell_state):
        embed_input = self.embedding(input)
        #forget gate's activation vector
        f = self.sigmoid(self.wf(embed_input) + self.uf(hidden))
        #input gate's activation vector
        i = self.sigmoid(self.wi(embed_input) + self.ui(hidden))
        #output gate's activation vector
        o = self.sigmoid(self.wo(embed_input) + self.uo(hidden))
        c_tilde = self.tanh(self.wc(embed_input) + self.uc(hidden))
        updated_cell_state = torch.mul(cell_state, f) + torch.mul(i, c_tilde)
        updated_hidden = torch.mul(self.tanh(updated_cell_state), o)
        output = self.softmax(self.out(updated_hidden))
        return output, updated_hidden, updated_cell_state

    def init_hidden(self):
        return Variable(torch.zeros(self.hidden_size))

    def init_cell_state(self):
        return Variable(torch.zeros(self.hidden_size))



class VocabularyLoader():
    def __init__(self, filename, device):
        self.character_table = {}
        self.index2char = {}
        self.n_chars = 0
        self.device = device
        with open(filename,'r',encoding='UTF-8') as f:
            lines=f.readlines()
            for line in lines:
                for w in line:
                    if w not in self.character_table:
                        self.character_table[w] = self.n_chars
                        self.index2char[self.n_chars] = w
                        self.n_chars += 1
        #print(self.n_chars)


    # Turn string into list of longs
    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor[c] = self.character_table[string[c]]
            except Exception as e:
                #pdb.set_trace()
                print(string[c])
                print(e)
        return Variable(tensor).to(self.device)


class DataLoader():
    def __init__(self, filename, chunk_len, device):
        with open(filename,'r',encoding='UTF-8') as f:
            lines=f.readlines()
        self.content = "".join(lines)
        self.file_len = len(self.content)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader(filename, self.device)


    def next_chunk(self):
        chunk = self.__random_chunk()
        input = chunk[:-1]
        target = chunk[1:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + chunk_len
        if(end_index > self.file_len):
            return self.vocabularyLoader.char_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.char_tensor(self.content[start_index:end_index])


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sys.argv.append('train')
    sys.argv.append('cre-utf8.txt')

    if(len(sys.argv)<2):
        print("usage: lstm [train file | inference (words vocabfile) ]")
        print("e.g. 1: lstm train cre-utf8.txt")
        print("e.g. 2: lstm inference words cre-utf8.txt")
        sys.exit(0)
    method = sys.argv[1]

    if(method == "train"):
        chunk_len = 100
        filename = sys.argv[2]
        dataloader = DataLoader(filename, chunk_len, device)
        trainer = Trainer(dataloader.vocabularyLoader.n_chars, device)
        trainer.train_model(dataloader)
    elif(method == "inference"):
        words = sys.argv[2]
        voc_file = sys.argv[3]
        inferencer = Inferencer("lstm_500000.model", voc_file, device)
        sentence = inferencer.getText(words,predict_len=100)
        print(sentence)
