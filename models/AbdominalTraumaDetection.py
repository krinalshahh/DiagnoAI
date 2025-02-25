import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class AbdominalTraumaDetection:
    def __init__(self, model_path, input_size=(224, 224), threshold=0.5, device=None):
        """
        Initialize the Abdominal Trauma Detection class.

        Args:
            model_path (str): Path to the pre-trained model weights.
            class_names (list): List of class names for multi-label classification.
            input_size (tuple): Input size for the model.
            threshold (float): Threshold for binary predictions (default: 0.5).
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = [
    'bowel_healthy', 'bowel_injury', 'extravasation_healthy', 'extravasation_injury',
    'kidney_healthy', 'kidney_low', 'kidney_high', 'liver_healthy', 'liver_low',
    'liver_high', 'spleen_healthy', 'spleen_low', 'spleen_high', 'any_injury'
]
        self.input_size = input_size
        self.threshold = threshold

        # Initialize and modify the EfficientNet model
        self.model = models.efficientnet_b0(weights=None)
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        self.model.classifier[1] = nn.Linear(
            in_features=self.model.classifier[1].in_features,
            out_features=len(self.class_names)
        )

        # Load the state dictionary
        state_dict = torch.load(model_path)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove prefix
        self.model.load_state_dict(new_state_dict)

        # Move the model to the selected device
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjust normalization if needed
        ])

    def infer(self, image_path):
        """
        Perform inference on an input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: A dictionary with class predictions and their confidence scores.
        """
        # Load and preprocess the image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            outputs = self.model(img)  # Forward pass
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()  # Apply sigmoid and move to CPU

        # Convert probabilities to binary predictions
        preds = (probs > self.threshold).astype(int)

        # Create a result dictionary
        result = {
            "predictions": {self.class_names[i]: preds[i] for i in range(len(self.class_names))},
            "confidences": {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        }

        return result

# Define class names

# Path to model weights
# model_path = "model/efficientnet_b0_Abdominal_Trauma_Detection.pth"
#
# # Initialize the detection class with a custom threshold
# trauma_detector = AbdominalTraumaDetection(model_path=model_path, threshold=0.7)
#
# # Perform inference
# image_path = "sampels/103.jpeg"
# result = trauma_detector.infer(image_path)
# print(result)
# # Display results
# print("Predictions:")
# for class_name, prediction in result["predictions"].items():
#     print(f"{class_name}: {prediction}")
#
# print("\nConfidence Scores:")
# for class_name, confidence in result["confidences"].items():
#     print(f"{class_name}: {confidence:.4f}")
