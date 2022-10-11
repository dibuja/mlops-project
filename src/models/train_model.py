import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from transformers import BertModel, BertTokenizer, AdamW,get_linear_schedule_with_warmup
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

from src.data.make_dataset import read_data

# TODO: Should be moved to .env file.
BATCH_SIZE = 16
MAX_LEN = 140
EPOCHS = 2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GenDataset(Dataset):
  def __init__(self, text, stars_review):
    self.text = text
    self.stars_review = stars_review
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    self.max_len = MAX_LEN
  
  def __len__(self):
    return len(self.text)
  
  def __getitem__(self, item):
    text = str(self.text[item])
    star = self.stars_review[item]

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'stars_review': torch.tensor(star, dtype=torch.long)
    }

class FineGrainedSentClassifier(nn.Module):

  def __init__(self, n_classes):
    super(FineGrainedSentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased',return_dict=False)
    self.drop = nn.Dropout(p=0.25)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

def create_data_loader(df):
  ds = GenDataset(
    text=df.text.to_numpy(),
    stars_review=df.stars_review.to_numpy()
  )

  return DataLoader(
    ds,
    batch_size=BATCH_SIZE
  )

def train_epoch(model, data_loader, loss_fn, optimizer, length):

  correct_predictions = 0
  p_loss = []

  model = model.train()

  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    stars_review = d["stars_review"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, stars_review)

    correct_predictions += torch.sum(preds == stars_review)
    p_loss.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

  return correct_predictions.double() / length, np.mean(p_loss)

def eval_model(model, data_loader, loss_fn, length):
  
  correct_predictions = 0
  p_loss = []

  model = model.eval()

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      stars_review = d["stars_review"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, stars_review)
      
      correct_predictions += torch.sum(preds == stars_review)
      p_loss.append(loss.item())

  return correct_predictions.double() / length, np.mean(p_loss)

def train():
    train, val, test = read_data()

    train_dl = create_data_loader(train)
    val_dl= create_data_loader(val)
    test_dl = create_data_loader(test)

    data = next(iter(train_dl))

    classes = sorted(train.stars_review.unique())

    model = FineGrainedSentClassifier(len(classes)).to(device)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=len(train_dl) * EPOCHS)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_acc = 0

    for count, epoch in enumerate(range(EPOCHS)):

        print('Epoch number:  '+str(count+1))
        print('----------------------------')

        train_acc, train_loss = train_epoch(model,train_dl,loss_fn, optimizer,len(train))

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(model,val_dl,loss_fn, len(val))

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc >= best_acc:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_acc = val_acc

    test_acc, _ = eval_model(model,test_dl,loss_fn,len(test))

    print('Accuracy: '+str(test_acc.item()))

if __name__ == '__main__':
    train()