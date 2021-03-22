import torch
import torchvision
import torchvision.transforms as transforms
from __future__ import print_function
from torch import nn, optim, cuda
import torch.nn.functional as F
import time
device = 'cuda' if cuda.is_available() else 'cpu' 

print('Training CIFAR10 Model on {:^30}\n{:}'.format(device,"============================================"))
trainset = torchvision.datasets.CIFAR10(root='/data',train=True,download=True,transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,num_workers=2)



testset = torchvision.datasets.CIFAR10(root='/data',train=False,download=True,transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=False,num_workers=2)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(32*32*3,740)#size를 찾아야함 12288=32*32*3*4
        self.l2 = nn.Linear(740, 520)
        self.l3 = nn.Linear(520, 300)
        self.l4 = nn.Linear(300, 100)
        self.l5 = nn.Linear(100, 10)
       
        

    def forward(self, x):
        x = x.view(-1,32*32*3)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))  
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        
        return self.l5(x) #위의 초기화와 forward함수를 통해 과정을 진행함


model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)#학습률 0.01 가속도 0.5



for epoch in range(10):
    running_loss = 0.0
    model.train()
    for i,(inputs,labels) in enumerate(trainloader):#i=batch size, ()는 데이터안의 인풋과 결과값
        inputs=inputs.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 500 == 0:
       
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(
                epoch, i * len(inputs), len(trainloader.dataset),
                100. * i / len(trainloader), loss.item())) #acc출력
    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device),labels.to(device)
            outputs = model(images)
            running_loss += loss.item()
            predicted = torch.max(outputs.data, 1)[1]
            
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / len(testloader.dataset)))

    
    
    
    
    
    
    Training CIFAR10 Model on              cuda             
============================================
Files already downloaded and verified
Files already downloaded and verified
Train Epoch: 0 | Batch Status: 0/50000 (0%) | Loss: 2.299480
Train Epoch: 0 | Batch Status: 32000/50000 (64%) | Loss: 2.004661
Accuracy of the network on the 10000 test images: 30 %
Train Epoch: 1 | Batch Status: 0/50000 (0%) | Loss: 1.729652
Train Epoch: 1 | Batch Status: 32000/50000 (64%) | Loss: 1.639654
Accuracy of the network on the 10000 test images: 34 %
Train Epoch: 2 | Batch Status: 0/50000 (0%) | Loss: 1.782162
Train Epoch: 2 | Batch Status: 32000/50000 (64%) | Loss: 1.513096
Accuracy of the network on the 10000 test images: 28 %
Train Epoch: 3 | Batch Status: 0/50000 (0%) | Loss: 2.156659
Train Epoch: 3 | Batch Status: 32000/50000 (64%) | Loss: 1.630726
Accuracy of the network on the 10000 test images: 42 %
Train Epoch: 4 | Batch Status: 0/50000 (0%) | Loss: 1.642429
Train Epoch: 4 | Batch Status: 32000/50000 (64%) | Loss: 1.488610
Accuracy of the network on the 10000 test images: 36 %
Train Epoch: 5 | Batch Status: 0/50000 (0%) | Loss: 1.882963
Train Epoch: 5 | Batch Status: 32000/50000 (64%) | Loss: 1.266753
Accuracy of the network on the 10000 test images: 43 %
Train Epoch: 6 | Batch Status: 0/50000 (0%) | Loss: 1.651680
Train Epoch: 6 | Batch Status: 32000/50000 (64%) | Loss: 1.256656
Accuracy of the network on the 10000 test images: 38 %
Train Epoch: 7 | Batch Status: 0/50000 (0%) | Loss: 1.725171
Train Epoch: 7 | Batch Status: 32000/50000 (64%) | Loss: 1.338931
Accuracy of the network on the 10000 test images: 48 %
Train Epoch: 8 | Batch Status: 0/50000 (0%) | Loss: 1.381072
Train Epoch: 8 | Batch Status: 32000/50000 (64%) | Loss: 1.482229
Accuracy of the network on the 10000 test images: 43 %
Train Epoch: 9 | Batch Status: 0/50000 (0%) | Loss: 1.450811
Train Epoch: 9 | Batch Status: 32000/50000 (64%) | Loss: 1.321882
Accuracy of the network on the 10000 test images: 47 %

    
    
    
    
