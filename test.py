import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=True)

# encapsulate them into dataloader form
# train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# class conv_net(nn.Module):
#     def __init__(self):
        

class MLP_net(nn.Module):
    def __init__(self, depth, dims, dropoutP):
        super().__init__()
        self.layer = []
        self.depth = depth
        self.dropout = dropoutP
        self.input = nn.Sequential(nn.Linear(28*28,dims[0]),
                            nn.ReLU(),
                            # nn.Dropout(self.dropout)
                            )
        for l in range(depth):
            if l<depth-1:
                self.layer.append( self.make_layer(dims[l],dims[l+1]))
        self.middle = nn.Sequential(*self.layer)
        self.output = nn.Sequential(nn.Linear(dims[-1],10),
                                nn.Softmax(dim=1))
        self.init_weights()

    def make_layer(self,input_ch, output_ch):
        l = nn.Sequential(nn.Linear(input_ch,output_ch),
                        nn.ReLU(),
                        nn.Dropout(self.dropout))
        return l

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                m.weight.data.normal_(0, 0.35)
                m.bias.data.zero_()

    def forward(self,x):
        output = self.input(x)
        output = self.middle(output)
        output = self.output(output)
        return output