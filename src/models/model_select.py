from .cnn_classifier import FmriCNNClassifier
from .multiview_classifier import FmriMultiviewClassifier

def select_model(model, width):
    if model == "cnn":
        return FmriCNNClassifier(width=width)
    elif model == "multiview":
        return FmriMultiviewClassifier(width=width)
    else:
        raise ValueError("Not available model")