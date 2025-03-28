import argparse
import torch
from utils import get_dataloader, seed_everything, evaluate_softvoting, evaluate_model
from models import select_model

def main(model, batch_size, width):
    """
    Run evaluation on trained model
    """
    seed_everything()  # Set random seed for reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model == "softvoting":
        # For soft voting, evaluate three separately trained models for each view
        views = ["axial", "coronal", "sagittal"]
        models = []
        loaders = []
        
        for i, view in enumerate(views):
            # Load test data for the specific view
            test_loader = get_dataloader(batch_size=batch_size, train=False, width=width, view_index=i)
            loaders.append(test_loader)
            
            # Load corresponding model weights
            model = select_model("cnn", in_channels=1, width=width)
            model_name = f"SoftVotingCNN_{width}_{view}"
            model_path = f"results/weights/{model_name}.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[INFO] Loaded model from {model_path}")
            models.append(model)
        
        # Evaluate with soft voting
        evaluate_softvoting(models, loaders, device)
    
    else:
        # Load test data for single model
        test_loader = get_dataloader(batch_size=batch_size, train=False, width=width)
        
        # Load model and weights
        model = select_model(model=model, in_channels=3, width=width)
        model_name = f"{model.__class__.__name__}_{width}"
        model_path = f"results/weights/{model_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded model from {model_path}")

        # Evaluate single model
        evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained fMRI classifier")
    
    # Argument definitions
    parser.add_argument("--model", type=str, default="cnn", help="Model class (default: cnn)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing (default: 16)")
    parser.add_argument("--width", type=int, default=256, help="Image width (default: 256)")

    args = parser.parse_args()
    main(model=args.model, batch_size=args.batch_size, width=args.width)
