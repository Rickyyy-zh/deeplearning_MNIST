import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse

from tqdm import tqdm
BATCH_SIZE = 8

def load_data():
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=True)

    # encapsulate them into dataloader form
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    return train_loader, test_loader
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
        x = x.view(-1,28 * 28)
        output = self.input(x)
        output = self.middle(output)
        output = self.output(output)
        return output


def main():
    # load_data()
    normalize = transforms.Normalize(mean=[.5], std=[.5])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=True)

    # encapsulate them into dataloader form
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    device = torch.device("cuda:2") if torch.cuda.is_available() else "cpu"
    print(device)
    model = MLP_net(6,[1024,2048,4096,2048,1024,10],0.1)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.7, weight_decay=0.0005)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_targets)
    print(example_data.shape)
    epochs=50
    ax_epoch = []
    ax_trainloss = []
    ax_testloss = []
    ax_acc = []
    train_num = len(train_loader)
    test_num = len(test_loader)
    print(train_num,test_num)
    for epoch in range(epochs):
        model.train()
        train_loss = 0 
        test_loss = 0
        for batch_idx, (input, target) in enumerate(train_loader):
            input=input.to(device)
            optimizer.zero_grad()
            out = model(input)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        acc =0.0
        best_acc = 0.0
        with torch.no_grad():
            for batch_idx, (test_in, target) in enumerate(test_loader):
                test_in = test_in.to(device)
                out = model(test_in)
                result = torch.argmax(out)
                loss = loss_fn(out, target)
                acc += torch.eq(result,target).sum().item()
                test_loss += loss.item()

        print("train epoch [{}/{}]  train_loss:{:.3f} test_loss:{:.3f} accurancy:{:.3f}".format(
                                    epoch+1, epochs, train_loss/train_num,test_loss/test_num, acc/test_num))
        if acc >= best_acc:
            best_acc = acc
            torch.save(model, "./best_pt.pt")
        torch.save(model,"./last_pt.pt")

        ax_trainloss.append( train_loss/train_num)
        ax_testloss.append(test_loss/test_num)
        ax_acc.append(acc/test_num)
        ax_epoch.append(epoch)
        # plot_curve(ax_epoch,ax_trainloss,ax_testloss,ax_acc)

    print("finshed training")


if __name__ == "__main__":
    main()
