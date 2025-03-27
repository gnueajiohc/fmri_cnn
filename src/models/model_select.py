from .cnn_classifier import FmriCNNClassifier
from .multiview_classifier import FmriMultiviewClassifier

def select_model(model, in_channels, width):
    if model == "cnn":
        return FmriCNNClassifier(in_channels=in_channels, width=width)
    elif model == "multiview":
        return FmriMultiviewClassifier(width=width)
    elif model == "softvoting":
        return None
    else:
        raise ValueError("Not available model")
    