import argparse
import torch
from utils import get_dataloader, seed_everything, train_single_model
from models import select_model

def train_model(model, train_loader, save_name, epochs, lr, device, softvoting=False):
    """
    Training model
    """
    if softvoting:
        views = ["axial", "coronal", "sagittal"]
        for i, view in enumerate(views):
            print(f"[SOFT-VOTING] Training view: {view}")
            
            # Get data loader for the specific view
            loader = get_dataloader(batch_size=train_loader.batch_size, train=True,
                                    width=train_loader.dataset.width, view_index=i)
            
            # Create a model for each view
            model_instance = select_model("cnn", in_channels=1, width=train_loader.dataset.width)
            inner_save_name = f"{save_name}_{view}"
            train_single_model(model_instance, loader, inner_save_name, epochs, lr, device)
    else:
        train_single_model(model, train_loader, save_name, epochs, lr, device)

def main(model, epochs, batch_size, lr, width):
    """
    Entry point for training a model
    """
    seed_everything()  # Set random seeds for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load training data
    train_loader = get_dataloader(batch_size=batch_size, train=True, width=width)
    
    # Select model architecture
    model = select_model(model, in_channels=3, width=width)

    # If model is None, use soft-voting with per-view CNNs
    if model is None:
        save_name = f"SoftVotingCNN_{width}"
        softvoting = True
    else:
        save_name = f"{model.__class__.__name__}_{width}"
        softvoting = False

    # Train the model(s)
    train_model(model, train_loader, save_name, epochs, lr, device, softvoting)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fMRI Classifier model trainer")
    
    # Argument parsing
    parser.add_argument("--model", type=str, default="cnn", help="Model class (default: cnn)")
    parser.add_argument("--epochs", type=int, default=10, help="Num of Epochs for training (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training (default: 1e-4)")
    parser.add_argument("--width", type=int, default=256, help="Image width (default: 256)")
    
    args = parser.parse_args()
    
    # Run main training routine
    main(model=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, width=args.width)
