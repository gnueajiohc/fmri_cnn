import torch.nn as nn

IN_CHANNELS = 3
NUM_CLASSES = 2

# -----------------------------------
# fMRI CNN Classifier Model Class
# -----------------------------------
class FmriCNNClassifier(nn.Module):
    """
    """
    def __init__(self, hidden_channels=[8, 16, 32], embedding_dimension=64, dropout_rate=0.4, width=256):
        super(FmriCNNClassifier, self).__init__()
        
        cnn_layers = []
        current_in_channels = IN_CHANNELS
        
        for current_out_channels in hidden_channels:
            cnn_layers.append(nn.Conv2d(current_in_channels,
                                    current_out_channels,
                                    stride=1,
                                    kernel_size=3,
                                    padding=1))
            cnn_layers.append(nn.SiLU())
            cnn_layers.append(nn.Dropout(p=dropout_rate))
            cnn_layers.append(nn.MaxPool2d(kernel_size=2))
            current_in_channels = current_out_channels
        
        self.cnn_layers = nn.Sequential(*cnn_layers)
        
        # after hidden layers size = (32, 32, 32)
        spatial_size = width // (2 ** len(hidden_channels))
        feature_size = current_in_channels * spatial_size * spatial_size
        
        self.out_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, embedding_dimension),
            nn.SiLU(),
            nn.Linear(embedding_dimension, NUM_CLASSES)
        )
        
    def forward(self, x):
        out = self.cnn_layers(x)
        return self.out_layers(out)
