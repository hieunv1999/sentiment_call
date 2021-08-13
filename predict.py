import torch
from models.LSTM import LSTMClassifier
import pandas as pd
import preprocess
from models.RCNN import RCNN
from models.selfAttention import SelfAttention
from tqdm import tqdm
from sklearn.metrics import classification_report
class Predict:
    def __init__(self,lenght_text):
        # self.TEXT, self.vocab_size, self.word_embeddings, self.train_iter = preprocess.get_datasets(32,1)
        checkpoint_file = 'checkpoint1.pth'
        checkpoint = torch.load(checkpoint_file)
        self.source = checkpoint['source']
        self.checkpoint = checkpoint['model_state_dict']
        self.vocab_size = len(self.source.vocab)
        self.word_embeddings = self.source.vocab.vectors
        self.batch_size = 1
        self.output_size = 3
        self.hidden_size = 256
        self.embedding_length = 300
        # self.model = LSTMClassifier(self.batch_size, self.output_size, self.hidden_size, self.vocab_size, self.embedding_length, self.word_embeddings)
        # self.model = RCNN(self.batch_size, self.output_size, self.hidden_size, self.vocab_size, self.embedding_length, self.word_embeddings)
        self.model = SelfAttention(self.batch_size, self.output_size, self.hidden_size, self.vocab_size, self.embedding_length, self.word_embeddings)
        self.model.load_state_dict(self.checkpoint)
        self.length_sent = lenght_text


    def predict(self,sentence):
        tokens = sentence.lower().split(' ')
        src_indexes = [self.source.vocab.stoi[to] for to in tokens]
        if len(src_indexes) < self.length_sent:
            src_indexes = src_indexes + (self.length_sent - len(src_indexes)) * [1]
        else:
            src_indexes = src_indexes[:self.length_sent]
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(1)
        src_tensor = torch.reshape(src_tensor, (1, self.length_sent))
        prediction = self.model(src_tensor)
        return torch.argmax(prediction).item()

    def evaluate(self):
        data = pd.read_excel('dataset/data_text_sentiment_vad_2.xlsx', sheet_name='TEST')
        sen = data['SENTENCE'].to_list()
        print(len(sen))
        label = data['LABEL'].to_list()
        pre = []
        for i in tqdm(range(len(sen))):
            pre.append(self.predict(sen[i]))
        target_names = ['Không cảnh báo', 'Cảnh báo thấp', 'Cảnh báo cao']
        print(classification_report(label, pre, target_names=target_names))
        dic = {'Sentence':sen,'Label':label,'predict':pre}
        dic = pd.DataFrame(dic)
        dic.to_excel('output_atten2_10.xlsx')
predict = Predict(32)
predict.evaluate()
# checkpoint_file = 'checkpoint1.pth'
# checkpoint = torch.load(checkpoint_file)
# TEXT = checkpoint['source']

# a = checkpoint['model_state_dict']


            
