import os
import argparse
import torch
import torch.nn as nn
from utils import get_dataloader, seed_everything
from models import FmriCNNClassifier

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

def main(batch_size, model_path):
    """
    테스트 실행
    """
    seed_everything()
    test_loader = get_dataloader(batch_size=batch_size, train=False)

    model = FmriCNNClassifier()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 파라미터 로드
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Loaded model from {model_path}")

    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained fMRI classifier")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing (default: 16)")
    parser.add_argument("--model_path", type=str, default="results/weights/fmri_cnn_classifier.pth",
                        help="Path to trained model weights")

    args = parser.parse_args()
    main(batch_size=args.batch_size, model_path=args.model_path)
