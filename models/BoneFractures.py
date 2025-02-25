import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class BoneFractures:
    def __init__(self, model_path, input_size=(224, 224), device=None):
        """
        Initialize the Bone Fractures Classification class.

        Args:
            model_path (str): Path to the pre-trained model weights.
            class_names (list): List of class names for classification.
            input_size (tuple): Input size for the model.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['fractured', 'not fractured']
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
            transforms.Resize(self.input_size),  # Pass input_size directly (not as a tuple)
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
            tuple: A tuple containing the predicted label, confidence score, and the processed image.
        """
        # Load and preprocess the image
        img = Image.open(image_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
            confidence, preds = torch.max(probabilities, 1)  # Get the highest confidence score and class index

            # Get the predicted label and confidence
            predicted_label = self.class_names[preds[0]]
            confidence_score = confidence[0].item()

            # Display the image and prediction
            #self.imshow_fixed_grid1(img_tensor.cpu().data[0], predicted_label)

            return {'predicted_label':predicted_label, 'confidence_score':confidence_score}


# # Define class names
#
# # Path to model weights
# model_path = "model/Bone_Fracture_Binary_Classification.pth"
#
# # Initialize the detection class
# classifier = BoneFractures(model_path=model_path)
#
# # Perform inference
# image_path = "sampels/photo_2024-12-08_23-34-37.jpg"
# result= classifier.infer(image_path)
# print(result)