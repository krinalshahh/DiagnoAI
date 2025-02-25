import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class KidneyDiseasesClassification:
    def __init__(self, model_path,  input_size=(224, 224), device=None):
        """
        Initialize the Kidney Diseases Classification class.

        Args:
            model_path (str): Path to the pre-trained model weights.
            class_names (list): List of class names for classification.
            input_size (tuple): Input size for the model.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']
        self.input_size = input_size

        # Initialize and modify the EfficientNet model
        self.model = models.efficientnet_b0()

        self.model.classifier[1] = nn.Linear(
            in_features=self.model.classifier[1].in_features,
            out_features=len(self.class_names)
        )

        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}  # Remove prefix
        self.model.load_state_dict(new_state_dict)

        # Move the model to the selected device
        self.model.to(self.device)
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def imshow_fixed_grid1(self, img, labels):
        """
        Displays an image with fixed grid size.

        Args:
            img (torch.Tensor): The input image tensor.
            labels (str): The label corresponding to the input image.
        """
        # Convert tensor to numpy array and adjust channel order
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))  # Change (C, H, W) to (H, W, C)

        # Reverse normalization to restore original color range
        npimg = npimg * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        npimg = np.clip(npimg, 0, 1)  # Clip values to valid range [0, 1]

        plt.imshow(npimg)
        plt.title(labels)
        plt.axis('off')
        plt.show()

    def infer(self, image_path):
        """
        Perform inference on an input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: Predicted label, confidence score, and processed image.
        """
        # Open the image
        img = Image.open(image_path)

        # Apply transformations and add batch dimension
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)

            predicted_label = self.class_names[preds[0]]
            confidence_score = confidence[0].item() * 100

            result = {
                "Predicted Label": predicted_label,
                "Confidence": f"{confidence_score:.2f}%"
            }

            # Display the image with the predicted label
            #self.imshow_fixed_grid1(img_tensor.cpu().data[0], f"{predicted_label} ({confidence_score:.2f}%)")

            return result


# Define class names

# # Path to model weights
# model_path = r"model/Kidney_Diseases_Classfication_Model.pth"
#
# # Initialize the detection class with a custom threshold
# classifier = KidneyDiseasesClassification(model_path=model_path)
#
# # Perform inference
# image_path = r"sampels/Cyst- (10).jpg"
# result = classifier.infer(image_path)
#
# # Print the result
# print(result)
