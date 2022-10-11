# TODO: Check that it runs correctly
# TODO: Generate a report in the right path
# TODO: load the model from the right path

import pandas as pd
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

from src.data.make_dataset import read_data

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

def create_data_loader(df):
  ds = GenDataset(
    text=df.text.to_numpy(),
    stars_review=df.stars_review.to_numpy()
  )

  return DataLoader(
    ds,
    batch_size=BATCH_SIZE
  )

def get_predictions(model, test_dl):
  model = model.eval()
  
  review_texts = []
  predictions = []
  real_values = []
  prediction_probs = []

  with torch.no_grad():
    for d in test_dl:

      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      stars_review = d["stars_review"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      probs = F.softmax(outputs, dim=1)

      review_texts+=texts
      predictions+=preds
      real_values+=stars_review
      prediction_probs+=probs

  return review_texts, torch.stack(predictions).cpu(), torch.stack(real_values).cpu(), torch.stack(prediction_probs).cpu()

def predict():

    # Data loading
    _, _, test = read_data()
    test_dl = create_data_loader(test)
    
    # Model loading
    model = torch.load('best_model_state.bin')

    y_review_texts, y_pred, y_test, y_pred_probs = get_predictions(model, test_dl)

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=[1,2,3,4,5], columns=[1,2,3,4,5])
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")


if __name__ == '__main__':
    predict()
