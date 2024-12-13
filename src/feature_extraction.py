import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()  
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image)
        return features.numpy()
