import torch
from torchvision import models, transforms
from PIL import Image
import os
import csv
from glob import glob

# --------- CONFIGURATION ---------
test_dir = "../Dataset_eld/train"  # Path to test images
model_path = "resnet18_final.pth"  # Path to trained ResNet18 model
num_classes = 7
class_names = ["1", "2", "3", "4", "5", "6", "7"]
save_csv = True
csv_path = "predictions_resnet.csv"
# ---------------------------------

# --------- IMAGE TRANSFORM ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# ----------------------------------

# --------- LOAD MODEL ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
# ------------------------------

# --------- PREDICTION LOOP ---------
all_images = glob(os.path.join(test_dir, "*", "*.jpg"))
correct = 0
total = 0
results = []

for image_path in all_images:
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
        
        true_class = os.path.basename(os.path.dirname(image_path))
        total += 1
        if predicted_class == true_class:
            correct += 1

        print(f"[{total}] Predicted: {predicted_class} | True: {true_class} | Match: {predicted_class == true_class}")
        results.append([os.path.basename(image_path), true_class, predicted_class])

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# --------- FINAL STATS ---------
accuracy = 100 * correct / total if total else 0
print(f"\nTotal images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")

# --------- SAVE CSV ---------
if save_csv:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "True Class", "Predicted Class"])
        writer.writerows(results)
    print(f"Predictions saved to: {csv_path}")
