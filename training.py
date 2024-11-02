from datetime import datetime

import pandas as pd
import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_training_and_test_data():
    df_training = pd.read_csv("artefacts/training_dataset.csv")

    all_features = []
    all_labels = []
    # Not sure that this is the right approach. Should we groupe by activity and then get features?
    for index, row in df_training.iterrows():
        prev = None if index == 0 else df_training.iloc[index - 1]
        super_prev = None if index < 2 else df_training.iloc[index - 2]
        features = [get_features(row, prev, super_prev)]
        label = [row['activity']]
        all_features.append(features)
        all_labels.append(label)

    return train_test_split(all_features, all_labels, test_size=0.2)

def get_features(row, prev, super_prev) -> dict:
    if prev is not None:
        date_str_trimmed = row['time'][:26]
        date_obj = datetime.strptime(date_str_trimmed, "%Y-%m-%d %H:%M:%S.%f")
        prev_date_str_trimmed = prev['time'][:26]
        prev_date_obj = datetime.strptime(prev_date_str_trimmed, "%Y-%m-%d %H:%M:%S.%f")
        time_delta = (date_obj - prev_date_obj).total_seconds()
    else:
        time_delta = 0

    return {
        # 'time': row['time'],
        # 'x': row['x'],
        # 'y': row['y'],
        # 'z': row['z'],
        # 'time_delta': time_delta,
        'dx': abs(row['x'] - prev['x']) if prev is not None else 0,
        'dy': abs(row['y'] - prev['y']) if prev is not None else 0,
        'dz': abs(row['z'] - prev['z']) if prev is not None else 0,
        # 'ddx': abs(row['x'] - super_prev['x']) if super_prev is not None else 0,
        # 'ddy': abs(row['y'] - super_prev['y']) if super_prev is not None else 0,
        # 'ddz': abs(row['z'] - super_prev['z']) if super_prev is not None else 0,
    }


def train_model():
    print("Training model")
    train_docs, test_docs, train_labels, test_labels = get_training_and_test_data()
    print("Training and test data retrieved")

    # Instantiate the trainer and set its parameters
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.set_params({
        'c1': 1,  # coefficient for L1 penalty
        'c2': .1,  # coefficient for L2 penalty
        'max_iterations': 5000,
        'feature.possible_transitions': True
    })

    print("Trainer instantiated and parameters set")

    # We are feeding our training set to the algorithm here.
    for xseq, yseq in zip(train_docs, train_labels):
        trainer.append(xseq, yseq)

    trainer.train('model.crfsuite')
    print("Model training completed")

def validate_model():
    print("Validating model")
    crf_tagger = pycrfsuite.Tagger()
    crf_tagger.open('model.crfsuite')
    print("Model opened")

    train_docs, test_docs, train_labels, test_labels = get_training_and_test_data()
    print("Training and test data retrieved")

    all_true, all_pred = [], []
    for index, x in enumerate(test_docs):
        all_true.extend(test_labels[index])
        predicted = crf_tagger.tag(x)
        all_pred.extend(predicted)

    print("Testing done")
    print(classification_report(all_true, all_pred))
