import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, AdamW,get_linear_schedule_with_warmup
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from src.data.make_dataset import read_data
import pickle

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import wandb

wandb.login()

wandb.init(project='project-mlops', entity="dtumlops36")

yaml_path = '/src/models/config/experiment/exp.yaml'
yaml_path = wandb.config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GenDataset(Dataset):
  """
  Creates the dataset of all reviews
  Arguments:
      text: a numpy array containing the review
      stars_review: a numpy array containing the stars given from 1-5
  Returns:
      The dataset used later by the data loader
  """

  def __init__(self, text, stars_review,model_type,max_len):
    self.text = text
    self.stars_review = stars_review
    self.tokenizer = BertTokenizer.from_pretrained(model_type)
    self.max_len = max_len
  
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
  """
  Creates the fine grained sentiment classifier based on the BERT MODEL
  Arguments:
      n_classes: an integer which defines the number of classes
  """

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

def create_data_loader(df,model_type,batch_size,max_len):
  """
  Uses the reviews (text and score) to create a Dataset for a later creation of a Data Loader which is going to be used in the training process
  """

  
  ds = GenDataset(
    text=df.text.to_numpy(),
    stars_review=df.stars_review.to_numpy(),
    model_type=model_type,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size
  )

def train_epoch(model, data_loader, loss_fn, optimizer, length):
  """
  Trains the model with the data previously stored in the dataloader
  Arguments:
      model: the bert model
      data_loader: a DataLoader with the dataset generated with the reviews
      loss_fn: a loss function
      optimizer: an optimizer (AdamW in that case)
      length: an integer defining the length of the dataset
  Returns:
      Two integers indicating the accuracy and loss of the model
  """

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
  """
  Evaluates the model with the test data
  Arguments:
      model: the bert model
      data_loader: a DataLoader with the dataset generated with the reviews
      loss_fn: a loss function
      length: an integer defining the length of the dataset
  Returns:
      An integer indicating the accuracy/loss of the model
  """
  
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
      
      wandb.log({"test_accuracy": correct_predictions.double() / length})

  return correct_predictions.double() / length, np.mean(p_loss)

@hydra.main(config_path="config", config_name="default_config.yaml")
def train():
    """
    Function used to train the model through different epoch and evaluate with the previous defined functions
    """

    cfg = cfg.experiment
    model_type = cfg.model
    epoch = cfg.hyper_param.epoch
    batch_size = cfg.hyper_param.batch_size
    max_len = cfg.max_len
    torch.manual_seed(cfg.seed)

    train, val, test = read_data()

    train_dl = create_data_loader(train,model_type,batch_size,max_len)
    val_dl= create_data_loader(val,model_type,batch_size,max_len)
    test_dl = create_data_loader(test,model_type,batch_size),max_len

    data = next(iter(train_dl))

    classes = sorted(train.stars_review.unique())

    model = FineGrainedSentClassifier(len(classes)).to(device)
    
    wandb.watch(model)

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.hyper_param.lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=len(train_dl) * epoch)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_acc = 0

    for count, epoch in enumerate(range(epoch)):

        print('Epoch number:  '+str(count+1))
        print('----------------------------')

        train_acc, train_loss = train_epoch(model,train_dl,loss_fn, optimizer,len(train))

        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        wandb.log({"epoch": epoch, "loss": train_loss})

        val_acc, val_loss = eval_model(model,val_dl,loss_fn, len(val))

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc >= best_acc:
            # Save to pickle file
            # TODO: Save the model directly in GCP
            pickle.dump(model, open('model.pkl', 'wb'))
            best_acc = val_acc

    test_acc, _ = eval_model(model,test_dl,loss_fn,len(test))

    print('Accuracy: '+str(test_acc.item()))

if __name__ == '__main__':
    train()