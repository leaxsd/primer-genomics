"""
Customized Pytorch training module.
"""
###############################################################
import torch
import numpy as np
from metrics import accuracy_score, precision_score, recall_score

###############################################################

# Training step function
def make_train_step(model, optimizer, criterion):
    """ Function that performs a step in the train loop.
    """
    def train_step(x, y):
        model.train() # set model to train mode
        optimizer.zero_grad() # zero gradients
        yhat = model(x) # make predictions
        loss = criterion(yhat, y) # evaluate loss
        loss.backward() # evaluate gradients
        optimizer.step() # update parameters
        return loss.item() # return loss value

    return train_step

class Train():
    """ Training class.
        Args:
            train_loader : torch.utils.data.DataLoader format input for training dataset.
            val_loader :  torch.utils.data.DataLoader format input for validation dataset.
            model : torch defined model.
            optimizer : torch.optim optimizer (e.g., torch.optim.Adam())
            criterion : torch.nn loss function (e.g., torch.nn.BCELoss())
            scheduler : learning rate scheduler (e.g., torch.optim.lr_scheduler.ReduceLROnPlateau()) [not required]
    """
    def __init__(self, train_loader, val_loader, model, optimizer, criterion, scheduler=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_step = make_train_step(self.model, optimizer, criterion)

    def step(self):
        """ Epochs step, training and validation.
            Return:
                training_loss, validation_loss, accuracy, precision, recall
        """
        # Training loop
        batch_loss, batch_val_loss, batch_accuracy, batch_precision, batch_recall = [], [], [], [], []

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            loss = self.train_step(x_batch, y_batch)
            batch_loss.append(loss)

        with torch.no_grad():
            # Validation loop
            for i, (x_val, y_val) in enumerate(self.val_loader):
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)

                self.model.eval()
                yhat = self.model(x_val)
                val_loss = self.criterion(yhat, y_val)
                batch_val_loss.append(val_loss)

                batch_accuracy.append(accuracy_score(np.argmax(yhat.cpu().detach().numpy(), axis=1), np.argmax(y_val.cpu().detach().numpy(), axis=1)))
                batch_precision.append(precision_score(np.argmax(yhat.cpu().detach().numpy(), axis=1), np.argmax(y_val.cpu().detach().numpy(), axis=1)))
                batch_recall.append(recall_score(np.argmax(yhat.cpu().detach().numpy(), axis=1), np.argmax(y_val.cpu().detach().numpy(), axis=1)))

        # step lr scheduler using val_loss
        if self.scheduler is not None:
            self.scheduler.step(val_loss)

        return [    torch.mean(torch.Tensor(batch_loss)),
                    torch.mean(torch.Tensor(batch_val_loss)),
                    torch.mean(torch.Tensor(batch_accuracy)),
                    torch.mean(torch.Tensor(batch_precision)),
                    torch.mean(torch.Tensor(batch_recall))
                ]
