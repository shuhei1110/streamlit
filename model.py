import torch
from torchvision import models, transforms
from PIL import Image

net = models.resnet101(pretrained=True)  
with open("imagenet_classes.txt") as f:  
    classes = [line.strip() for line in f.readlines()]

def predict(img):
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]
                                        )
                                    ])
    img = transform(img)
    x = torch.unsqueeze(img, 0)  

    net.eval()
    y = net(x)

    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  
    return [(classes[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
