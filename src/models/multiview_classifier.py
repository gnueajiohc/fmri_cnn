import torch
import torch.nn as nn
from .cnn_classifier import FmriCNNClassifier

NUM_CLASSES = 2

# -----------------------------------
# fMRI Multiview Classifier Model Class
# -----------------------------------
class FmriMultiviewClassifier(nn.Module):
    """
    """
    def __init__(self, feature_dim=64, hidden_dim=64, dropout_rate=0.4, width=256):
        super(FmriMultiviewClassifier, self).__init__()
        self.axial_cnn = FmriCNNClassifier(in_channels=1, embedding_dimension=feature_dim, dropout_rate=dropout_rate, width=width, full=False)
        self.coronal_cnn = FmriCNNClassifier(in_channels=1, embedding_dimension=feature_dim, dropout_rate=dropout_rate, width=width, full=False)
        self.sagittal_cnn = FmriCNNClassifier(in_channels=1, embedding_dimension=feature_dim, dropout_rate=dropout_rate, width=width, full=False)
        
        total_feature_dim = 3 * feature_dim
        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(total_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )
    
    def forward(self, x):
        # x shape: [B, 3, 256, 256]
        axial = x[:, 0:1, :, :]
        coronal = x[:, 1:2, :, :]
        sagittal = x[:, 2:3, :, :]

        a_feat = self.axial_cnn(axial)
        c_feat = self.coronal_cnn(coronal)
        s_feat = self.sagittal_cnn(sagittal)

        feat = torch.cat([a_feat, c_feat, s_feat], dim=1)  # [B, 3*feature_dim]
        return self.classifier(feat)
    