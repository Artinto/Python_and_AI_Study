from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import torchvision
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

transforms_train = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                      transforms.ToTensor()])

train_data_set = torchvision.datasets.ImageFolder(root = '/content/drive/MyDrive/ramen/train',transform=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=16, shuffle=True)

test_data_set = torchvision.datasets.ImageFolder(root = '/content/drive/MyDrive/ramen/test',transform=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=16, shuffle=True)

print(len(train_data_set), len(test_data_set))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 4)

    def forward(self, x):
        in_size = x.size(0)
        #print(x.shape)
        x = F.relu(self.pool(self.conv1(x))) 
        x = F.relu(self.pool(self.conv2(x)))
        #print(x.shape)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net()
img.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(img.parameters(),lr=0.005,momentum=0.9)\

for epoch in range(5):
    loss_sum = 0
    correct1 = 0
    total1 = 0
    for i, data in enumerate(trainloader,0):
        inputs,labels = data[0].to(device),data[1].to(device)
        optimizer.zero_grad()
        outputs = img(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, 1)
        total1 += labels.size(0)
        correct1 += (pred==labels).sum().item()
        loss_sum += loss.item()
        if i % 50 == 0:    # print every 2000 mini-batches
            print('[%d, %1d] loss: %.3f' %(epoch + 1, i + 1, loss_sum / 50))
            loss_sum = 0.0
    print('epoch %d Accuracy: %d %%' %(epoch+1, 100*correct1/total1))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device),data[1].to(device)
        outputs = img(images)
        _, pred1 = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred1 == labels).sum().item()

print('Accuracy of test images: %d %%' % (100 * correct / total))

#결과
1949 100
[1, 1] loss: 0.001
[1, 51] loss: 0.075
[1, 101] loss: 0.071
epoch 1 Accuracy: 96 %
[2, 1] loss: 0.002
[2, 51] loss: 0.077
[2, 101] loss: 0.096
epoch 2 Accuracy: 97 %
[3, 1] loss: 0.001
[3, 51] loss: 0.078
[3, 101] loss: 0.197
epoch 3 Accuracy: 94 %
[4, 1] loss: 0.002
[4, 51] loss: 0.118
[4, 101] loss: 0.123
epoch 4 Accuracy: 95 %
[5, 1] loss: 0.001
[5, 51] loss: 0.103
[5, 101] loss: 0.186
epoch 5 Accuracy: 95 %
Accuracy of test images: 86 %
