#!/usr/bin/env python
###############################################
import os
import torch
import pandas as pd
import numpy as np
import argparse
import requests
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from model import Net
from train import Train

###############################################
#  Args
###############################################

ap = argparse.ArgumentParser()

# input sequences url
ap.add_argument('--SEQUENCES_URL', type=str, default='https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/sequences.txt',
                    help='URL for input sequences')
# input labels url
ap.add_argument('--LABELS_URL', type=str, default='https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/labels.txt',
                    help='URL for input labels')
# save folder
# ap.add_argument('--DATA_FOLDER', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/'),
#                     help="Saved model folder")
ap.add_argument('--DATA_FOLDER', type=str, default='data/',
                    help="Saved model folder")

# training settings
ap.add_argument('-ep', '--epochs', type=int, default=50,
                    help='Total number od training epochs')

ap.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                    help='Define optimizer learing rate')

ap.add_argument('-p', '--patience', type=int, default=10,
                    help='Scheduler patiance epochs')

ap.add_argument('-sf', '--scheduler_factor', type=float, default=0.01,
                    help='factor on which loss should reduce within patiance epochs')

ap.add_argument('-mlr', '--min_lr', type=float, default=0.00001,
                    help='Minimum learning rate')

ap.add_argument('-sm', '--save_model', type=bool, default=True,
                    help="Save the best model")

ap.add_argument('--test_size', type=float, default=0.25,
                    help="Percent of input data set as Test")

ap.add_argument('--val_size', type=float, default=0.25,
                    help="Percent of train data set as validation")

ap.add_argument('-bs', '--batch_size', type=int, default=32,
                    help="Training batch size")

args = ap.parse_args()

###############################################

def run(args):
    """
        Run set training test.
    """
    # Load model
    model = Net()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Set criterion (Loss Function)
    criterion = torch.nn.BCELoss()

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=False, patience=args.patience, factor=args.scheduler_factor, min_lr=args.min_lr)

    # Load sequences
    sequences = requests.get(args.SEQUENCES_URL).text.split('\n')
    sequences = list(filter(None, sequences))  # This removes empty sequences.

    # %% The LabelEncoder encodes a sequence of bases as a sequence of integers.
    integer_encoder = LabelEncoder()
    # The OneHotEncoder converts an array of integers to a sparse matrix
    one_hot_encoder = OneHotEncoder(categories='auto')
    input_features = []

    for sequence in sequences:
      integer_encoded = integer_encoder.fit_transform(list(sequence))
      integer_encoded = np.array(integer_encoded).reshape(-1, 1)
      one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
      input_features.append(one_hot_encoded.toarray())
    input_features = np.stack(input_features)

    # Load labels
    labels = requests.get(args.LABELS_URL).text.split('\n')
    labels = list(filter(None, labels))  # removes empty sequences
    one_hot_encoder = OneHotEncoder(categories='auto')
    input_labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(input_labels).toarray()

    # Split data
    input_features = input_features.transpose(0,2,1).astype(np.float32) # adapt to pytorch input format [N, C, L]
    input_labels = input_labels.astype(np.float32) # adapt to pytorch input format
    train_features, test_features, train_labels, test_labels = train_test_split(input_features, input_labels, test_size=args.test_size, random_state=42)
    train_features, val_features, train_labels, val_labels =  train_test_split(train_features, train_labels, test_size=args.val_size, random_state=42)

    # Set pytorch dataloader
    train_loader = torch.utils.data.DataLoader(list(zip(train_features, train_labels)), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(list(zip(val_features, val_labels)), batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(list(zip(test_features, test_labels)), batch_size=1, shuffle=False, num_workers=2)

    # Declare training class
    train = Train(train_loader, val_loader, model, optimizer, criterion, scheduler)

    # make save dirs
    os.makedirs(args.DATA_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(args.DATA_FOLDER, 'models/'), exist_ok=True)

    # main loop
    train_loss, val_loss, accuracy, precision, recall = [], [], [], [], []
    for epoch in tqdm(range(args.epochs), desc="Training"):
        train_loss_, val_loss_, accuracy_, precision_, recall_ = train.step()
        train_loss.append(train_loss_)
        val_loss.append(val_loss_)
        accuracy.append(accuracy_)
        precision.append(precision_)
        recall.append(recall_)

        # save models
        if val_loss_ <= min(val_loss):
            torch.save(model.state_dict(), os.path.join(args.DATA_FOLDER, 'models/best_model.pt'))
        elif epoch == (args.epochs-1):
            torch.save(model.state_dict(), os.path.join(args.DATA_FOLDER, 'models/last_model.pt'))

    # save data
    for name, data in ({'train_loss': train_loss, 'val_loss': val_loss,'accuracy' : accuracy, 'precision' : precision, 'recall' : recall}).items():
        np.save(os.path.join(args.DATA_FOLDER, '{}.npy'.format(name)), data)

    # Predict test data
    model.load_state_dict(torch.load(os.path.join(args.DATA_FOLDER, 'models/best_model.pt')));
    model.eval();
    test_pred = [np.argmax(model(x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu().detach().numpy(), axis=1) for x, _ in test_loader]
    test_label = [np.argmax(y.cpu().detach().numpy(), axis=1) for _, y in test_loader]
    np.save(os.path.join(args.DATA_FOLDER, 'test_pred.npy'), test_pred)
    np.save(os.path.join(args.DATA_FOLDER, 'test_label.npy'), test_label)

###############################################
# Main
###############################################
if __name__ == '__main__':
    run(args)
