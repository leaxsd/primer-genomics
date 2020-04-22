################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import requests
import torch
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from model import Net
from train import Train
from torchsummary import summary
from tqdm import tqdm
################################################

SEQUENCES_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/sequences.txt'

sequences = requests.get(SEQUENCES_URL).text.split('\n')
sequences = list(filter(None, sequences))  # This removes empty sequences.

# Let's print the first few sequences.
pd.DataFrame(sequences, index=np.arange(1, len(sequences)+1), columns=['Sequences']).head()

# %% The LabelEncoder encodes a sequence of bases as a sequence of integers.
integer_encoder = LabelEncoder()
# The OneHotEncoder converts an array of integers to a sparse matrix where
# each row corresponds to one possible value of each feature.
one_hot_encoder = OneHotEncoder(categories='auto')
input_features = []

for sequence in sequences:
  integer_encoded = integer_encoder.fit_transform(list(sequence))
  integer_encoded = np.array(integer_encoded).reshape(-1, 1)
  one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
  input_features.append(one_hot_encoded.toarray())

np.set_printoptions(threshold=40)
input_features = np.stack(input_features)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
print('Classes: ', integer_encoder.classes_)
print('One hot encoding of Sequence #1:\n',input_features[0].T)
input_features.shape

# %% Read Labels
LABELS_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/labels.txt'

labels = requests.get(LABELS_URL).text.split('\n')
labels = list(filter(None, labels))  # removes empty sequences
one_hot_encoder = OneHotEncoder(categories='auto')
labels = np.array(labels).reshape(-1, 1)
input_labels = np.array(labels)

input_labels = one_hot_encoder.fit_transform(labels).toarray()

print('Labels:\n',labels.T)
print('One-hot encoded labels:\n',input_labels.T)


# %% Split data
input_features = input_features.transpose(0,2,1).astype(np.float32) # adapt to pytorch input format [N, C, L]
input_features.shape
input_labels = input_labels.astype(np.float32) # adapt to pytorch input format
input_labels.shape
train_features, test_features, train_labels, test_labels = train_test_split(input_features, input_labels, test_size=0.25, random_state=42)
train_features, val_features, train_labels, val_labels =  train_test_split(train_features, train_labels, test_size=0.25, random_state=42)

# Declare pytorch imported model
net = Net().to('cuda')
summary(net, input_size=input_features.shape[-2:])


# %% training model in pytorch

train_loader = torch.utils.data.DataLoader(list(zip(train_features, train_labels)), batch_size=4, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(list(zip(val_features, val_labels)), batch_size=4, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(list(zip(test_features, test_labels)), batch_size=1, shuffle=False, num_workers=2)

train = Train(train_loader, val_loader)

train_loss, val_loss, accuracy, precision, recall = [], [], [], [], []

for epoch in tqdm(range(50)):
    train_loss_, val_loss_, accuracy_, precision_, recall_ = train.step()
    train_loss.append(train_loss_)
    val_loss.append(val_loss_)
    accuracy.append(accuracy_)
    precision.append(precision_)
    recall.append(recall_)

# %% plot loss
plt.figure(figsize=(8,6))
plt. plot(train_loss, label="Training loss")
plt. plot(val_loss, label="Validation loss")
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.ylim(-0.01,0.2)
plt.legend(frameon=False)
plt.plot();

# %% plot accuracy
plt.figure(figsize=(8,6))
plt. plot(accuracy, label="validation accuracy")
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(frameon=False)
plt.plot();

# %% test saved model
from metrics import num_score

model = Net()
model.load_state_dict(torch.load('data/models/best_model.pt'));
model.eval();

preds = [np.argmax(model(x).cpu().detach().numpy(), axis=1) for x, _ in test_loader]
labels = [np.argmax(y.cpu().detach().numpy(), axis=1) for _, y in test_loader]

FP, FN, TP, TN = num_score(np.array(preds), np.array(labels))
print('FP', FP, 'FN', FN, 'TP', TP, 'TN', TN)

# %%
cm = np.array([[TN, FP], [FN, TP]]) / 250.
df_cm = pd.DataFrame(cm, range(2), range(2))
plt.figure(figsize=(8,6))
plt.title('Normalized confusion matrix')
sns.set(font_scale=1.25) # for label size
sns.heatmap(df_cm.round(2), annot=True, cmap="Blues", fmt='.2')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.plot();

# %% saliency maps
