import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

def seed_everything(seed=42):
    """ seed every random choices """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[INFO] Setting random seed to {seed}')

def train_single_model(model, train_loader, save_name, epochs, lr, device):
    """
    Train a single model with given training data and save the weights.
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

def evaluate_softvoting(models, loaders, device):
    """
    model evaluate function for softvoting model, print accuracy
    """
    for model in models:
        model.to(device)
        model.eval()
        
    correct = 0
    total = 0
    
    # weights
    alpha = [0.33, 0.33, 0.33]
    
    with torch.no_grad():
        for batch_axial, batch_coronal, batch_sagittal in zip(*loaders):
            # image from each view
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
            
            # weighted average
            avg_prob = alpha[0] * prob1 + alpha[1] * prob2 + alpha[2] * prob3
            _, predicted = torch.max(avg_prob, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    accuracy = correct / total * 100
    print(f"\n[TEST RESULT] Accuracy: {accuracy:.2f}% ({correct}/{total})")
            

def evaluate_model(model, test_loader, device):
    """
    model evaluate function, print accuracy
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # image from data
            images = images.to(device)
            labels = labels.to(device)

            # output from model
            logits = model(images)
            _, predicted = torch.max(logits, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"\n[TEST RESULT] Accuracy: {accuracy:.2f}% ({correct}/{total})")