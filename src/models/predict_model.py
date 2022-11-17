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
import pickle

import hydra
from hydra.utils import to_absolute_path

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

def create_data_loader(df,batch_size,model_type,max_len):
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

def get_predictions(model, test_dl):
  """
  Gets predictions for the test data used for the trained model
  Arguments:
      model: the trained model used to get predictions
      test_dl: the dataset used to get predictions
  Returns:
      The numeric rating predictions for the given reviews
  """

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


@hydra.main(config_path="config", config_name="default_config.yaml")
def predict():
    """
    Utilises the previously defined functions to make the predictions.
    """

    cfg = cfg.experiment
    model_type = cfg.model
    batch_size = cfg.hyper_param.batch_size
    max_len = cfg.max_len
    torch.manual_seed(cfg.seed)

    # Data loading
    _, _, test = read_data()
    test_dl = create_data_loader(test,batch_size,model_type,max_len)
    
    # Model loading
    model = pickle.load(open('model.pkl', 'rb'))

    y_review_texts, y_pred, y_test, y_pred_probs = get_predictions(model, test_dl)
  
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=[1,2,3,4,5], columns=[1,2,3,4,5])
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")


if __name__ == '__main__':
    predict()
