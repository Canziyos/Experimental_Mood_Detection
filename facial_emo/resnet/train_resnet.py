import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import os

def main():
    # ----------- CONFIGURATION -----------
    dataset_path = "H:/Experimental_Mood_Detection/data/DATASET/train"
    num_classes = len(os.listdir(dataset_path))
    batch_size = 8
    num_epochs = 25
    learning_rate = 1e-4
    # -------------------------------------

    # ----------- TRANSFORMATIONS -----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # ---------------------------------------

    # ----------- DATASET LOADING -----------
    train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    # ---------------------------------------

    # ----------- MODEL SETUP -----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    # -----------------------------------

    # ----------- LOSS & OPTIMIZER -----------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # ----------------------------------------

    # ----------- TRAINING LOOP -----------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=running_loss / (total / batch_size), acc=100 * correct / total)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} complete | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}%")

        torch.save(model.state_dict(), f"resnet18_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), "resnet18_final.pth")
    print("Final model saved to resnet18_final.pth")

# For multiprocessing on Windows
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
