import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as PILImage

class Knee_Osteoporosis:
    def __init__(self):
        self.class_labels = ['Normal', 'Osteopenia', 'Osteoporosis']  # Adjusted for 3 classes

    def preprocess_image(self, image_path):
        # Define image preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load the image
        image = PILImage.open(image_path)
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return input_tensor


    def inf(self, model_weights_path, image_path):
        # Set device to GPU if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, len(self.class_labels)) 
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model = model.to(device)
        model.eval()

        # Preprocess the image
        input_tensor = self.preprocess_image(image_path).to(device)

        # Run inference
        with torch.no_grad():
            # Perform inference
            output = model(input_tensor)
            predictions = torch.softmax(output, dim=1)
            class_index = torch.argmax(predictions).item()
            confidence = predictions[0][class_index].item()

        # Print results
        # print(f"Predicted Class: {self.class_labels[class_index]}")
        # print(f"Confidence: {confidence:.4f}")
        return {'predicted_label': self.class_labels[class_index], 'confidence_score': confidence}


# # Example usage
# classifier = Knee_Osteoporosis()
# model_weights_path = r"model/Knee_model_weights.pth"
# image_path = r"KneeSamples\Normal\photo_1_2024-12-18_02-57-19.jpg"
#
# result=classifier.inf(model_weights_path, image_path)
