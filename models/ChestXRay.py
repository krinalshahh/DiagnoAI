import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import warnings

warnings.filterwarnings("ignore")

class ChestXRay:
    def __init__(self, model_path, input_size=(224, 224), device=None):
        """
        Initialize the Chest X-Ray Classification class.

        Args:
            model_path (str): Path to the pre-trained model weights.
            class_names (list): List of class names for classification.
            input_size (tuple): Input size for the model.
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['NORMAL', 'PNEUMONIA bacteria', 'PNEUMONIA virus']
        self.input_size = input_size

        # Initialize and modify the EfficientNet model
        self.model = models.efficientnet_b0()

        self.model.classifier[1] = nn.Linear(
            in_features=self.model.classifier[1].in_features,
            out_features=len(self.class_names)
        )

        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=self.device)
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
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Assuming img is in CHW format
        plt.title(labels)  # Using labels to set title of the plot
        plt.axis('off')
        plt.show()

    def infer(self, image_path):
        """
        Perform inference on an input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: Predicted label and confidence score as text.
        """

        img = Image.open(image_path)
        img = self.transform(img).unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)

            # Display predicted label
            ax = plt.subplot(2, 2, 1)
            ax.axis('off')
            ax.set_title(f'Predicted: {self.class_names[preds[0]]} ({confidence.item() * 100:.2f}%)')

            # Pass the predicted label to imshow_fixed_grid1
            #self.imshow_fixed_grid1(img.cpu().data[0], self.class_names[preds[0]])
        return {'predicted_label': self.class_names[preds[0]], 'confidence_score': confidence.item()}

# Define class names

# Path to model weights
# model_path = "model/chest_xray.pth"
#
# # Initialize the detection class with a custom threshold
# classifier = ChestXRay(model_path=model_path)
#
# # Perform inference
# image_path = "sampels/BACTERIA-2034017-0006.jpeg"
# result = classifier.infer(image_path)
#
# print(result)
