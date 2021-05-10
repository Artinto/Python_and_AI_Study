import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

num_classes = 7
input_size = 7 
hidden_size = 7 
batch_size = 1 
sequence_length = 9 
num_layers = 1 

idx2char = ['h','e','l','o','w','r','d']

x_data = [0,1,2,2,3,4,3,5,2] #x : helloworl
one_hot_lookup = [[1,0,0,0,0,0,0], 
                  [0,1,0,0,0,0,0],
                  [0,0,1,0,0,0,0],
                  [0,0,0,1,0,0,0],
                  [0,0,0,0,1,0,0],
                  [0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,1]]

x_one_hot = [one_hot_lookup[x] for x in x_data]
y_data = [1,2,2,3,4,3,5,2,6] #y : elloworld

inputs = torch.Tensor(x_one_hot)
labels = torch.LongTensor(y_data)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.rnn=nn.RNN(input_size = input_size,hidden_size=hidden_size,batch_first=True)
        
    def forward(self,x,hidden):
        x=x.view(batch_size,sequence_length,input_size)
        
        out,hidden = self.rnn(x,hidden)
        
        out = out.view(-1,num_classes)
        return out
    
    def init_hidden(self):
        return torch.zeros(num_layers,batch_size,hidden_size)

model = Model()

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(),lr = 0.01)

for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    cor = ""
    result_string = "" 
    hidden = model.init_hidden() 

    outputs = model(inputs,hidden)
    loss = criterion(outputs,labels)

    _, idx = outputs.max(1)
    result_str = [idx2char[c] for c in idx.squeeze()]
    for i in result_str:
        result_string += i
    if result_string == "elloworld":
        cor = "Correct"
    else:
        cor = "Wrong"
    print("epoch: %d, loss: %1.3f" % (epoch+1,loss))
    print("Predicted string: ", ''.join(result_str), cor)
    loss.backward()
    optimizer.step()
    
    #결과
    epoch: 1, loss: 1.950
Predicted string:  drrrdrddr Wrong
epoch: 2, loss: 1.917
Predicted string:  lrrrdrrrr Wrong
epoch: 3, loss: 1.885
Predicted string:  lrrrdrrrr Wrong
epoch: 4, loss: 1.855
Predicted string:  lrrrdrrrr Wrong
epoch: 5, loss: 1.827
Predicted string:  lrrrdrror Wrong
epoch: 6, loss: 1.799
Predicted string:  lrrrdrror Wrong
epoch: 7, loss: 1.773
Predicted string:  lrrrdrror Wrong
epoch: 8, loss: 1.748
Predicted string:  lrrrdrror Wrong
epoch: 9, loss: 1.724
Predicted string:  lrrrdrror Wrong
epoch: 10, loss: 1.701
Predicted string:  llrroorod Wrong
epoch: 11, loss: 1.678
Predicted string:  lllloorld Wrong
epoch: 12, loss: 1.657
Predicted string:  lllloorld Wrong
epoch: 13, loss: 1.636
Predicted string:  lllloorld Wrong
epoch: 14, loss: 1.615
Predicted string:  lllloorld Wrong
epoch: 15, loss: 1.595
Predicted string:  lllloorld Wrong
epoch: 16, loss: 1.576
Predicted string:  lllloorld Wrong
epoch: 17, loss: 1.557
Predicted string:  lllloorld Wrong
epoch: 18, loss: 1.539
Predicted string:  lllloorld Wrong
epoch: 19, loss: 1.521
Predicted string:  lllloorld Wrong
epoch: 20, loss: 1.503
Predicted string:  lllloorld Wrong
epoch: 21, loss: 1.486
Predicted string:  lllloorld Wrong
epoch: 22, loss: 1.469
Predicted string:  lllloorld Wrong
epoch: 23, loss: 1.452
Predicted string:  lllloorld Wrong
epoch: 24, loss: 1.436
Predicted string:  lllooorld Wrong
epoch: 25, loss: 1.419
Predicted string:  lllooorld Wrong
epoch: 26, loss: 1.403
Predicted string:  lllooorld Wrong
epoch: 27, loss: 1.387
Predicted string:  lllooorld Wrong
epoch: 28, loss: 1.370
Predicted string:  lllooorld Wrong
epoch: 29, loss: 1.354
Predicted string:  lllooorld Wrong
epoch: 30, loss: 1.339
Predicted string:  lllooorld Wrong
epoch: 31, loss: 1.323
Predicted string:  lllooorld Wrong
epoch: 32, loss: 1.307
Predicted string:  lllooorld Wrong
epoch: 33, loss: 1.292
Predicted string:  lllooorld Wrong
epoch: 34, loss: 1.277
Predicted string:  lllooorld Wrong
epoch: 35, loss: 1.262
Predicted string:  lllooorld Wrong
epoch: 36, loss: 1.247
Predicted string:  lllooorld Wrong
epoch: 37, loss: 1.233
Predicted string:  lllooorld Wrong
epoch: 38, loss: 1.219
Predicted string:  lllooorld Wrong
epoch: 39, loss: 1.205
Predicted string:  lllooorld Wrong
epoch: 40, loss: 1.192
Predicted string:  lllooorld Wrong
epoch: 41, loss: 1.179
Predicted string:  lllooorld Wrong
epoch: 42, loss: 1.166
Predicted string:  ellooorld Wrong
epoch: 43, loss: 1.153
Predicted string:  ellooorld Wrong
epoch: 44, loss: 1.140
Predicted string:  elloworld Correct
epoch: 45, loss: 1.128
Predicted string:  elloworld Correct
epoch: 46, loss: 1.116
Predicted string:  elloworld Correct
epoch: 47, loss: 1.104
Predicted string:  elloworld Correct
epoch: 48, loss: 1.092
Predicted string:  elloworld Correct
epoch: 49, loss: 1.081
Predicted string:  elloworld Correct
epoch: 50, loss: 1.070
Predicted string:  elloworld Correct
epoch: 51, loss: 1.059
Predicted string:  elloworld Correct
epoch: 52, loss: 1.048
Predicted string:  elloworld Correct
epoch: 53, loss: 1.038
Predicted string:  elloworld Correct
epoch: 54, loss: 1.029
Predicted string:  elloworld Correct
epoch: 55, loss: 1.019
Predicted string:  elloworld Correct
epoch: 56, loss: 1.010
Predicted string:  elloworld Correct
epoch: 57, loss: 1.002
Predicted string:  elloworld Correct
epoch: 58, loss: 0.993
Predicted string:  elloworld Correct
epoch: 59, loss: 0.985
Predicted string:  elloworld Correct
epoch: 60, loss: 0.978
Predicted string:  elloworld Correct
epoch: 61, loss: 0.971
Predicted string:  elloworld Correct
epoch: 62, loss: 0.964
Predicted string:  elloworld Correct
epoch: 63, loss: 0.957
Predicted string:  elloworld Correct
epoch: 64, loss: 0.950
Predicted string:  elloworld Correct
epoch: 65, loss: 0.944
Predicted string:  elloworld Correct
epoch: 66, loss: 0.938
Predicted string:  elloworld Correct
epoch: 67, loss: 0.932
Predicted string:  elloworld Correct
epoch: 68, loss: 0.926
Predicted string:  elloworld Correct
epoch: 69, loss: 0.920
Predicted string:  elloworld Correct
epoch: 70, loss: 0.915
Predicted string:  elloworld Correct
epoch: 71, loss: 0.910
Predicted string:  elloworld Correct
epoch: 72, loss: 0.905
Predicted string:  elloworld Correct
epoch: 73, loss: 0.900
Predicted string:  elloworld Correct
epoch: 74, loss: 0.895
Predicted string:  elloworld Correct
epoch: 75, loss: 0.890
Predicted string:  elloworld Correct
epoch: 76, loss: 0.886
Predicted string:  elloworld Correct
epoch: 77, loss: 0.882
Predicted string:  elloworld Correct
epoch: 78, loss: 0.878
Predicted string:  elloworld Correct
epoch: 79, loss: 0.874
Predicted string:  elloworld Correct
epoch: 80, loss: 0.870
Predicted string:  elloworld Correct
epoch: 81, loss: 0.866
Predicted string:  elloworld Correct
epoch: 82, loss: 0.862
Predicted string:  elloworld Correct
epoch: 83, loss: 0.858
Predicted string:  elloworld Correct
epoch: 84, loss: 0.855
Predicted string:  elloworld Correct
epoch: 85, loss: 0.852
Predicted string:  elloworld Correct
epoch: 86, loss: 0.848
Predicted string:  elloworld Correct
epoch: 87, loss: 0.845
Predicted string:  elloworld Correct
epoch: 88, loss: 0.842
Predicted string:  elloworld Correct
epoch: 89, loss: 0.839
Predicted string:  elloworld Correct
epoch: 90, loss: 0.836
Predicted string:  elloworld Correct
epoch: 91, loss: 0.833
Predicted string:  elloworld Correct
epoch: 92, loss: 0.830
Predicted string:  elloworld Correct
epoch: 93, loss: 0.827
Predicted string:  elloworld Correct
epoch: 94, loss: 0.824
Predicted string:  elloworld Correct
epoch: 95, loss: 0.822
Predicted string:  elloworld Correct
epoch: 96, loss: 0.819
Predicted string:  elloworld Correct
epoch: 97, loss: 0.817
Predicted string:  elloworld Correct
epoch: 98, loss: 0.814
Predicted string:  elloworld Correct
epoch: 99, loss: 0.812
Predicted string:  elloworld Correct
epoch: 100, loss: 0.809
Predicted string:  elloworld Correct
