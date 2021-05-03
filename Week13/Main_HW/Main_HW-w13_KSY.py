from __future__ import print_function
import argparse
import os
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from torch.autograd import Variable


class CustomImageDataset(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length

hyper_param_epoch = 20
hyper_param_batch = 24
hyper_param_learning_rate = 0.001

transforms_train = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

transforms_test = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()])
train_data_set = CustomImageDataset(data_set_path="./CUSTOM/train", transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)

test_data_set = CustomImageDataset(data_set_path="./CUSTOM/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=False)

class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
def trainst(epoch):
    custom_model.train()
  
    for i_batch, item in enumerate(train_loader):
            images = item['image'].to(device)
            labels = item['label'].to(device)

            # Forward pass
            outputs = custom_model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch + 1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, hyper_param_epoch, loss.item()))
def test():
    custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for item in test_loader:
            images = item['image'].to(device)
            labels = item['label'].to(device)
            outputs = custom_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    
        
    

for epoch in range(hyper_param_epoch):
    trainst(epoch)
    test()
    
    
    
    
    
    
Epoch [1/20], Loss: 1.2187
Epoch [1/20], Loss: 1.1451
Epoch [1/20], Loss: 0.8994
Epoch [1/20], Loss: 0.8604
Epoch [1/20], Loss: 0.8512
Epoch [1/20], Loss: 0.9435
Epoch [1/20], Loss: 0.7851
Epoch [1/20], Loss: 0.7094
Test Accuracy of the model on the 100 test images: 72.0 %
Epoch [2/20], Loss: 0.6712
Epoch [2/20], Loss: 0.6395
Epoch [2/20], Loss: 0.7089
Epoch [2/20], Loss: 0.5786
Epoch [2/20], Loss: 0.5715
Epoch [2/20], Loss: 0.5933
Epoch [2/20], Loss: 0.6970
Epoch [2/20], Loss: 0.5254
Test Accuracy of the model on the 100 test images: 86.0 %
Epoch [3/20], Loss: 0.6319
Epoch [3/20], Loss: 0.5236
Epoch [3/20], Loss: 0.5311
Epoch [3/20], Loss: 0.7002
Epoch [3/20], Loss: 0.6107
Epoch [3/20], Loss: 0.5925
Epoch [3/20], Loss: 0.5376
Epoch [3/20], Loss: 0.6083
Test Accuracy of the model on the 100 test images: 91.0 %
Epoch [4/20], Loss: 0.4634
Epoch [4/20], Loss: 0.3985
Epoch [4/20], Loss: 0.5517
Epoch [4/20], Loss: 0.4121
Epoch [4/20], Loss: 0.4390
Epoch [4/20], Loss: 0.3485
Epoch [4/20], Loss: 0.3637
Epoch [4/20], Loss: 0.3381
Test Accuracy of the model on the 100 test images: 86.0 %
Epoch [5/20], Loss: 0.3978
Epoch [5/20], Loss: 0.3695
Epoch [5/20], Loss: 0.3882
Epoch [5/20], Loss: 0.3783
Epoch [5/20], Loss: 0.3011
Epoch [5/20], Loss: 0.3112
Epoch [5/20], Loss: 0.3713
Epoch [5/20], Loss: 0.4983
Test Accuracy of the model on the 100 test images: 76.0 %
Epoch [6/20], Loss: 0.2999
Epoch [6/20], Loss: 0.4254
Epoch [6/20], Loss: 0.4350
Epoch [6/20], Loss: 0.2587
Epoch [6/20], Loss: 0.3203
Epoch [6/20], Loss: 0.4094
Epoch [6/20], Loss: 0.4650
Epoch [6/20], Loss: 0.3448
Test Accuracy of the model on the 100 test images: 87.0 %
Epoch [7/20], Loss: 0.2580
Epoch [7/20], Loss: 0.3500
Epoch [7/20], Loss: 0.3053
Epoch [7/20], Loss: 0.2945
Epoch [7/20], Loss: 0.2628
Epoch [7/20], Loss: 0.2014
Epoch [7/20], Loss: 0.2847
Epoch [7/20], Loss: 0.2760
Test Accuracy of the model on the 100 test images: 91.0 %
Epoch [8/20], Loss: 0.3935
Epoch [8/20], Loss: 0.2637
Epoch [8/20], Loss: 0.4281
Epoch [8/20], Loss: 0.2759
Epoch [8/20], Loss: 0.4242
Epoch [8/20], Loss: 0.3008
Epoch [8/20], Loss: 0.2562
Epoch [8/20], Loss: 0.3046
Test Accuracy of the model on the 100 test images: 88.0 %
Epoch [9/20], Loss: 0.2408
Epoch [9/20], Loss: 0.2241
Epoch [9/20], Loss: 0.2877
Epoch [9/20], Loss: 0.2097
Epoch [9/20], Loss: 0.2470
Epoch [9/20], Loss: 0.1876
Epoch [9/20], Loss: 0.2109
Epoch [9/20], Loss: 0.1661
Test Accuracy of the model on the 100 test images: 88.0 %
Epoch [10/20], Loss: 0.3428
Epoch [10/20], Loss: 0.2642
Epoch [10/20], Loss: 0.2299
Epoch [10/20], Loss: 0.2680
Epoch [10/20], Loss: 0.1518
Epoch [10/20], Loss: 0.1798
Epoch [10/20], Loss: 0.2414
Epoch [10/20], Loss: 0.2642
Test Accuracy of the model on the 100 test images: 91.0 %
Epoch [11/20], Loss: 0.1377
Epoch [11/20], Loss: 0.2502
Epoch [11/20], Loss: 0.1484
Epoch [11/20], Loss: 0.2036
Epoch [11/20], Loss: 0.1727
Epoch [11/20], Loss: 0.1849
Epoch [11/20], Loss: 0.2131
Epoch [11/20], Loss: 0.2730
Test Accuracy of the model on the 100 test images: 90.0 %
Epoch [12/20], Loss: 0.2037
Epoch [12/20], Loss: 0.1911
Epoch [12/20], Loss: 0.1682
Epoch [12/20], Loss: 0.1892
Epoch [12/20], Loss: 0.2569
Epoch [12/20], Loss: 0.1428
Epoch [12/20], Loss: 0.1643
Epoch [12/20], Loss: 0.2650
Test Accuracy of the model on the 100 test images: 89.0 %
Epoch [13/20], Loss: 0.1735
Epoch [13/20], Loss: 0.1187
Epoch [13/20], Loss: 0.1895
Epoch [13/20], Loss: 0.1634
Epoch [13/20], Loss: 0.1204
Epoch [13/20], Loss: 0.1548
Epoch [13/20], Loss: 0.1920
Epoch [13/20], Loss: 0.1224
Test Accuracy of the model on the 100 test images: 89.0 %
Epoch [14/20], Loss: 0.1700
Epoch [14/20], Loss: 0.2531
Epoch [14/20], Loss: 0.1921
Epoch [14/20], Loss: 0.1463
Epoch [14/20], Loss: 0.1480
Epoch [14/20], Loss: 0.1227
Epoch [14/20], Loss: 0.1753
Epoch [14/20], Loss: 0.1720
Test Accuracy of the model on the 100 test images: 93.0 %
Epoch [15/20], Loss: 0.1190
Epoch [15/20], Loss: 0.1543
Epoch [15/20], Loss: 0.2141
Epoch [15/20], Loss: 0.1331
Epoch [15/20], Loss: 0.1788
Epoch [15/20], Loss: 0.1639
Epoch [15/20], Loss: 0.1287
Epoch [15/20], Loss: 0.0984
Test Accuracy of the model on the 100 test images: 88.0 %
Epoch [16/20], Loss: 0.1290
Epoch [16/20], Loss: 0.0901
Epoch [16/20], Loss: 0.1041
Epoch [16/20], Loss: 0.2565
Epoch [16/20], Loss: 0.1568
Epoch [16/20], Loss: 0.1360
Epoch [16/20], Loss: 0.1356
Epoch [16/20], Loss: 0.1321
Test Accuracy of the model on the 100 test images: 90.0 %
Epoch [17/20], Loss: 0.3092
Epoch [17/20], Loss: 0.1324
Epoch [17/20], Loss: 0.2322
Epoch [17/20], Loss: 0.1031
Epoch [17/20], Loss: 0.2038
Epoch [17/20], Loss: 0.1331
Epoch [17/20], Loss: 0.1314
Epoch [17/20], Loss: 0.1950
Test Accuracy of the model on the 100 test images: 95.0 %
Epoch [18/20], Loss: 0.1713
Epoch [18/20], Loss: 0.1150
Epoch [18/20], Loss: 0.1166
Epoch [18/20], Loss: 0.1678
Epoch [18/20], Loss: 0.0781
Epoch [18/20], Loss: 0.1181
Epoch [18/20], Loss: 0.0790
Epoch [18/20], Loss: 0.0926
Test Accuracy of the model on the 100 test images: 92.0 %
Epoch [19/20], Loss: 0.2188
Epoch [19/20], Loss: 0.1447
Epoch [19/20], Loss: 0.1440
Epoch [19/20], Loss: 0.1239
Epoch [19/20], Loss: 0.0754
Epoch [19/20], Loss: 0.1744
Epoch [19/20], Loss: 0.0832
Epoch [19/20], Loss: 0.1156
Test Accuracy of the model on the 100 test images: 95.0 %
Epoch [20/20], Loss: 0.0800
Epoch [20/20], Loss: 0.2242
Epoch [20/20], Loss: 0.1030
Epoch [20/20], Loss: 0.1035
Epoch [20/20], Loss: 0.1317
Epoch [20/20], Loss: 0.0918
Epoch [20/20], Loss: 0.0955
Epoch [20/20], Loss: 0.2049
Test Accuracy of the model on the 100 test images: 93.0 %
