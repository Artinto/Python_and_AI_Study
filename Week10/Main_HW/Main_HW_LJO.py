import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

class Img(nn.Module):
    def __init__(self):
        super(Img,self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

img = Img()
img.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(img.parameters(),lr=0.001,momentum=0.9)

for epoch in range(3):
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
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, loss_sum / 2000))
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
[1,  2000] loss: 2.174
[1,  4000] loss: 1.804
[1,  6000] loss: 1.632
[1,  8000] loss: 1.563
[1, 10000] loss: 1.479
[1, 12000] loss: 1.442
epoch 1 Accuracy: 38 %
[2,  2000] loss: 1.373
[2,  4000] loss: 1.338
[2,  6000] loss: 1.317
[2,  8000] loss: 1.304
[2, 10000] loss: 1.264
[2, 12000] loss: 1.270
epoch 2 Accuracy: 53 %
[3,  2000] loss: 1.198
[3,  4000] loss: 1.183
[3,  6000] loss: 1.172
[3,  8000] loss: 1.158
[3, 10000] loss: 1.155
[3, 12000] loss: 1.159
epoch 3 Accuracy: 58 %
Accuracy of test images: 59 %
