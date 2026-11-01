import torch.nn as nn
import torch.nn.functional as F

class FCN_tune(nn.Module):
    def __init__(self, in_channels=1,
        # input=500,
        num_classes=2,
        filters=(128, 256, 128),
        kernels=(8, 5, 3),
        dropouts=(0.0, 0.0, 0.0)):
        super(FCN_tune, self).__init__()

        # self.metrics = {"epochs": [],
        #                 "accuracy": [],
        #                 "recall": [],
        #                 "precision": [],
        #                 "F1": [],
        #                 "train_loss": [],
        #                 "val_loss": []}

        # self.layers = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=8, stride=1, padding="same"),
        #     nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding="same"),
        #     nn.BatchNorm1d(256, eps=0.001, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same"),
        #     nn.BatchNorm1d(128, eps=0.001, momentum=0.01),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        #     nn.Linear(in_features=128, out_features=num_classes)
        # )

        self.layers = nn.Sequential(
            # First convolutional block
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters[0],
                kernel_size=kernels[0],
                stride=1,
                padding="same"
            ),
            nn.BatchNorm1d(
                filters[0], eps=0.001, momentum=0.01
            ),
            nn.ReLU(),
            nn.Dropout(dropouts[0]),

            # Second convolutional block
            nn.Conv1d(
                in_channels=filters[0],
                out_channels=filters[1],
                kernel_size=kernels[1],
                stride=1,
                padding="same"
            ),
            nn.BatchNorm1d(
                filters[1], eps=0.001, momentum=0.01
            ),
            nn.ReLU(),
            nn.Dropout(dropouts[1]),

            # Third convolutional block
            nn.Conv1d(
                in_channels=filters[1],
                out_channels=filters[2],
                kernel_size=kernels[2],
                stride=1,
                padding="same"
            ),
            nn.BatchNorm1d(
                filters[2], eps=0.001, momentum=0.01
            ),
            nn.ReLU(),
            nn.Dropout(dropouts[2]),

            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            # Classifier
            nn.Linear(
                in_features=filters[2],
                out_features=num_classes
            )
        )
        
    def forward(self, x):
        return self.layers(x)

    # def record_metrics(self, epochs, accuracy, recall, precision, 
    #                    F1, train_loss, val_loss):
    #     self.metrics["epochs"].append(epochs)
    #     self.metrics["accuracy"].append(accuracy)
    #     self.metrics["recall"].append(recall)
    #     self.metrics["precision"].append(precision)
    #     self.metrics["F1"].append(F1)
    #     self.metrics["train_loss"].append(train_loss)
    #     self.metrics["val_loss"].append(val_loss)