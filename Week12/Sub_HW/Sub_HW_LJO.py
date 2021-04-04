from PIL import Image
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
from torchvision import transforms

tf = transforms.Compose([transforms.Resize((100,100)), transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

modelset = torchvision.datasets.ImageFolder(root = '/content/drive/MyDrive/origin_data',transform=tf)

trainset, testset = torch.utils.data.random_split(modelset,[400, 151])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=0)

print(len(trainset), len(testset))

for data in testloader:
    images, labels = data
    plt.imshow(torchvision.utils.make_grid(images, nrow=4, normalize=True).permute(1,2,0))
    plt.show()
    break
