import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_channels = 1, input=500, num_classes=2):
        super(FCN, self).__init__()

        self.metrics = {"epochs": [],
                        "accuracy": [],
                        "recall": [],
                        "precision": [],
                        "F1": [],
                        "train_loss": [],
                        "val_loss": []}

        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=8, stride=1, padding="same"),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same"),
            nn.BatchNorm1d(256, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(128, eps=0.001, momentum=0.99),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

    def record_metrics(self, epochs, accuracy, recall, precision, 
                       F1, train_loss, val_loss):
        self.metrics["epochs"].append(epochs)
        self.metrics["accuracy"].append(accuracy)
        self.metrics["recall"].append(recall)
        self.metrics["precision"].append(precision)
        self.metrics["F1"].append(F1)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)