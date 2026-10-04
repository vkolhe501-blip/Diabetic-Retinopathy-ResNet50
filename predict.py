"""
Diabetic Retinopathy Detection using ResNet-50
Inference script to evaluate retinal fundus images.
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Class definitions based on the 3-class retinopathy dataset
CLASS_NAMES = [
    "Class 0 (No Diabetic Retinopathy)",
    "Class 1 (Mild / Moderate Diabetic Retinopathy)",
    "Class 2 (Severe / Proliferative Diabetic Retinopathy)"
]

# Standard ImageNet normalization and 224x224 input sizing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(weights_path="best_retinopathy_resnet50.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    # Build ResNet-50 with 3 output classes
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, device


def predict_image(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted_idx = int(torch.argmax(outputs, dim=1)[0].cpu().item())

    return predicted_idx, probabilities


def main():
    parser = argparse.ArgumentParser(description="Predict Diabetic Retinopathy from Retinal Fundus Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input retinal image")
    parser.add_argument("--weights", type=str, default="best_retinopathy_resnet50.pt", help="Path to model weights")
    args = parser.parse_args()

    print(f"Loading model weights from: {args.weights} ...")
    model, device = load_model(args.weights)
    print(f"Model loaded successfully on {device}.")

    predicted_idx, probabilities = predict_image(args.image, model, device)

    print("\n" + "=" * 50)
    print(f"PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted Class: {CLASS_NAMES[predicted_idx]}")
    print(f"Confidence:      {probabilities[predicted_idx] * 100:.2f}%\n")
    print("Class Probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "█" * int(probabilities[i] * 30)
        print(f"  {name:50s} : {probabilities[i] * 100:6.2f}% | {bar}")
    print("=" * 50)


if __name__ == "__main__":
    main()
