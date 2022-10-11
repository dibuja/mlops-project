# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data():
    """ Runs data processing scripts to turn raw data from (../raw) into
        the three datasets: train, test and val.
    """

    input_filepath = 'data/raw/raw.csv'

    # Read data from ../raw
    df = pd.read_csv(input_filepath)

    # Map stars from 0-4 to remove errors
    df['stars_review'] = df['stars_review'].apply(lambda x: x-1)

    # Process the data
    train, test = train_test_split(df, test_size=0.2)
    val, test = train_test_split(test, test_size=0.5)

    return train, val, test

if __name__ == "__main__":
    pass