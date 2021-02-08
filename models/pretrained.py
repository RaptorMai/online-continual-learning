import torchvision.models as models
import torch

def ResNet18_pretrained(n_classes):
    classifier = models.resnet18(pretrained=True)
    classifier.fc = torch.nn.Linear(512, n_classes)
    return classifier