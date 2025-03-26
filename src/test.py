import os
import argparse
import torch
import torch.nn as nn
from utils import get_dataloader, seed_everything
from models import select_model

def evaluate_softvoting(models, loaders, device):
    """
    """
    for model in models:
        model.to(device)
        model.eval()
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_axial, batch_coronal, batch_sagittal in zip(*loaders):
            images1, labels = batch_axial
            images2, _      = batch_coronal
            images3, _      = batch_sagittal
            
            images1 = images1.to(device)
            images2 = images2.to(device)
            images3 = images3.to(device)
            labels  = labels.to(device)
            
            logits1 = models[0](images1)
            logits2 = models[1](images2)
            logits3 = models[2](images3)
            
            prob1 = torch.softmax(logits1, dim=1)
            prob2 = torch.softmax(logits2, dim=1)
            prob3 = torch.softmax(logits3, dim=1)
            
            avg_prob = (prob1 + prob2 + prob3) / 3
            _, predicted = torch.max(avg_prob, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total * 100
    print(f"\n[TEST RESULT] Accuracy: {accuracy:.2f}% ({correct}/{total})")
            

def evaluate_model(model, test_loader, device):
    """
    모델 평가 함수 - Accuracy만 출력
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, predicted = torch.max(logits, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"\n[TEST RESULT] Accuracy: {accuracy:.2f}% ({correct}/{total})")

def main(model, batch_size, width):
    """
    테스트 실행
    """
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model == "softvoting":
        views = ["axial", "coronal", "sagittal"]
        models = []
        loaders = []
        
        for i, view in enumerate(views):
            # load data
            test_loader = get_dataloader(batch_size=batch_size, train=False, width=width, view_index=i)
            loaders.append(test_loader)
            
            # load model parameters
            model = select_model("cnn", in_channels=1, width=width)
            model_name = f"SoftVotingCNN_{width}_{view}"
            model_path = f"results/weights/{model_name}.pth"
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[INFO] Loaded model from {model_path}")
            models.append(model)
    else:
        # load data
        test_loader = get_dataloader(batch_size=batch_size, train=False, width=width)
        
        # load model parameters
        model = select_model(model=model, width=width)
        model_name = f"{model.__class__.__name__}_{width}"
        model_path = f"results/weights/{model_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded model from {model_path}")

        evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained fMRI classifier")
    
    parser.add_argument("--model", type=str, default="cnn", help="Model class (default: cnn)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing (default: 16)")
    parser.add_argument("--width", type=int, default=256, help="Image width (default: 256)")

    args = parser.parse_args()
    main(model=args.model, batch_size=args.batch_size, width=args.width)
