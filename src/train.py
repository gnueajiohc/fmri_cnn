import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloader, seed_everything
from models import select_model

def train_model(model, train_loader, save_name, epochs, lr, device, softvoting=False):
    if softvoting:
        views = ["axial", "coronal", "sagittal"]
        for i, view in enumerate(views):
            print(f"[SOFT-VOTING] Training view: {view}")
            
            loader = get_dataloader(batch_size=train_loader.batch_size, train=True,
                                    width=train_loader.dataset.width, view_index=i)
            
            model_instance = select_model("cnn", in_channels=1, width=train_loader.dataset.width)
            inner_save_name = f"{save_name}_{view}"
            _train_single_model(model_instance, loader, inner_save_name, epochs, lr, device)
    else:
        _train_single_model(model, train_loader, save_name, epochs, lr, device)
        

def _train_single_model(model, train_loader, save_name, epochs, lr, device):
    """
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print("[TRAINING]".center(50, '-'))
    print("")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        avg_loss = total_loss / total
        accuracy = correct / total * 100
        
        print(f"[Epoch {epoch + 1} of {epochs}] Loss: {avg_loss:.5f}  Acurracy: {accuracy:.2f}%")
    
        # calculating training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n[INFO] Total training time: {elapsed_time:.2f} seconds")
    
    # save trained model parameters
    save_dir = "results/weights"
    os.makedirs(save_dir, exist_ok=True)
    model_path = f"{save_dir}/{save_name}.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"[INFO] Model saved at {model_path}\n")

def main(model, epochs, batch_size, lr, width):
    """
    """
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loader = get_dataloader(batch_size=batch_size, train=True, width=width)
    model = select_model(model, in_channels=3, width=width)

    
    if model is None:
        save_name = f"SoftVotingCNN_{width}"
        softvoting = True
    else:
        save_name = f"{model.__class__.__name__}_{width}"
        softvoting = False
    train_model(model, train_loader, save_name, epochs, lr, device, softvoting)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fMRI Classifier model trainer")
    
    parser.add_argument("--model", type=str, default="cnn", help="Model class (default: cnn)")
    parser.add_argument("--epochs", type=int, default=10, help="Num of Epochs for training (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training (default: 1e-4)")
    parser.add_argument("--width", type=int, default=256, help="Image width (default: 256)")
    
    args = parser.parse_args()
    
    main(model=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, width=args.width)
    