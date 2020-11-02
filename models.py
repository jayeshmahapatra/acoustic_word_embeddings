import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self,num_output):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv1d(40,96,(10))
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(96, 96, (8))
        self.pool2 = nn.MaxPool1d(3)
        #self.fc1 = nn.Linear(1728, 1024)
        self.fc1 = nn.Linear(672, 1024)
        self.fc2 = nn.Linear(1024, num_output)
        self.sm = nn.Softmax(dim = 1)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print('Pre')
        #print(x.shape)  
        x = x.view(x.shape[0], -1)
        #print('Post')
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = F.log_softmax(x,dim=1)
        #print(x.shape)
        #print("Done")
        return x
    
    def give_embeddings(self,x,dev):
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print('Pre')
        #print(x.shape)  
        x = x.view(x.shape[0], -1)
        #print('Post')
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        return x.cpu().detach().numpy() if dev.type == 'cuda' else x.detach().numpy()


class SiameseNet(nn.Module):
    def __init__(self, dim_out = 1024):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv1d(40,96,(10))
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(96, 96, (8))
        self.pool2 = nn.MaxPool1d(3)
        #self.fc1 = nn.Linear(1728, 1024)
        self.fc1 = nn.Linear(672, dim_out)
    
    def forward(self, x):
        
        x = self.pool1(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        #print('Pre')
        #print(x.shape)  
        x = x.view(x.shape[0], -1)
        #print('Post')
        #print(x.shape)
        x = self.fc1(x)
        
        return x
    
    def give_embeddings(self,x,dev):
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print('Pre')
        #print(x.shape)  
        x = x.view(x.shape[0], -1)
        #print('Post')
        #print(x.shape)
        x = F.relu(self.fc1(x))
        return x.cpu().detach().numpy() if dev.type == 'cuda' else x.detach().numpy()

class OrthographicNet(nn.Module):
    def __init__(self,num_input,num_output):
        super(OrthographicNet, self).__init__()
        self.fc1 = nn.Linear(num_input, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_output)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = F.relu(self.fc3(x))
        #print(x.shape)
        return x
    
    def give_embeddings(self,x,dev):
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        x = F.relu(self.fc3(x))
        #print(x.shape)
        #print("Done")
        return x.cpu().detach().numpy() if dev.type == 'cuda' else x.detach().numpy()


