import preprocess
import torch
import torch.nn.functional as F
from models.LSTM import LSTMClassifier
from models.RCNN import RCNN
from models.selfAttention import SelfAttention
class Train:
    def __init__(self,batch_size, output_size, hidden_size, embedding_length):
        self.batch_size = batch_size
        self.learning_rate = 2e-5
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length
        self.TEXT, self.vocab_size, self.word_embeddings, self.train_iter , self.val_iter = preprocess.get_datasets(32,self.batch_size)
        # self.model = LSTMClassifier(self.batch_size, self.output_size, self.hidden_size, self.vocab_size, self.embedding_length, self.word_embeddings)
        # self.model = RCNN(self.batch_size, self.output_size, self.hidden_size, self.vocab_size, self.embedding_length, self.word_embeddings)
        self.model = SelfAttention(self.batch_size, self.output_size, self.hidden_size, self.vocab_size, self.embedding_length, self.word_embeddings)
        self.loss_fn = F.cross_entropy
    def clip_gradient(self,model, clip_value):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)
    def train_model(self, model, train_iter, epoch):
        total_epoch_loss = 0
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        steps = 0
        model.train()
        for idx, (text,label) in enumerate(train_iter):
            optim.zero_grad()
            prediction = model(text)
            loss = self.loss_fn(prediction, label)
            loss.backward()
            self.clip_gradient(model, 1e-1)
            optim.step()
            steps += 1

            if steps % 5 == 0:
                print(
                    f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}')
            total_epoch_loss += loss.item()
        return total_epoch_loss / len(train_iter)
    def valid_model(self, model, valid_iter, epoch):
        total_epoch_loss = 0
        steps = 0
        model.eval()
        for idx, (text,label) in enumerate(valid_iter):
            prediction = model(text)
            loss = self.loss_fn(prediction, label)
            steps += 1
            if steps % 5 == 0:
                print(
                    f'Epoch: {epoch + 1}, Idx: {idx + 1}, Valid Loss: {loss.item():.4f}')
            total_epoch_loss += loss.item()
        return total_epoch_loss / len(valid_iter)
    def training(self):
        for epoch in range(20):
            train_loss = self.train_model(self.model, self.train_iter, epoch)
            val_loss = self.valid_model(self.model, self.val_iter, epoch)
            if epoch%3==0:
                torch.save({
                    'source': self.TEXT,
                    'model_state_dict': self.model.state_dict(),
                    }, 'checkpoint{}.pth'.format(epoch+1))
            print(
                f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
# batch_size = 256
# learning_rate = 2e-5
# output_size = 3
# hidden_size = 256
# embedding_length = 300
# TEXT, vocab_size, word_embeddings, train_iter = preprocess.get_datasets(32,batch_size)
# model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
# for text,label in train_iter:
#     model(text)
train = Train(batch_size=256, output_size=3, hidden_size=256, embedding_length=300)
data_train = train.training()
