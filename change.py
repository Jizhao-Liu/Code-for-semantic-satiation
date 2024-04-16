import  torch
import torchvision
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms,utils
from    torch import nn, optim
import numpy as np
import torchvision.models as models
import  skimage
from skimage import io,transform,util
from PIL import Image
import matplotlib.pyplot as plt
#plt.ion()
from    ccnn import INIT,CONV,FC
from visdom import Visdom

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

unloader = transforms.ToPILImage()


def main():

    batchsz=200
    dataset0 = datasets.MNIST('../data', train = False, download = True, 
                            transform=transforms.Compose([transforms.Resize((28,28)),
                                                        transforms.ToTensor()
                                                        ]))
    # Selecting classes 
    idx0 = (dataset0.targets==1) 
    dataset0.targets = dataset0.targets[idx0]
    dataset0.data = dataset0.data[idx0]

    print(dataset0.data.size())
    dataset0 = DataLoader(dataset0,batch_size = batchsz,shuffle=True,drop_last=True)


    dataset1 = datasets.MNIST('../data', train = False, download = True, 
                            transform=transforms.Compose([transforms.Resize((28,28)),
                                                        transforms.ToTensor()
                                                        ]))
    # Selecting classes 
    idx1 = (dataset1.targets==9) 
    dataset1.targets = dataset1.targets[idx1]
    dataset1.data = dataset1.data[idx1]
    print(dataset1.data.size())
    dataset1 = DataLoader(dataset1,batch_size = batchsz,shuffle=True,drop_last=True)
    device = torch.device('cuda')
    model3 = torch.load('net1.pth')
    model1 = torch.load('net2.pth')
    model2 = torch.load('net3.pth')
    viz=Visdom()
    model3.eval()
    model1.eval()
    model2.eval()

    with torch.no_grad():
    # test
        a=10

        aa=100-a
        models = [model3] + [model1] * a
        total_correct = [0] * a
        total_num = [0] * a
        
        for x, label in dataset0:
            x, label = x.to(device), label.to(device)
            f, l, u, e, y = model3(x)
            features = [(f, l, u, e, y)]
            
            for i in range(a):
                f, l, u, e, y = model1(x, *features[-1])
                features.append((f, l, u, e, y))
                
                logit = model2(y)
                pred = logit.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct[i] += correct
                total_num[i] += x.size(0)
                
        acc1 = [c / n for c, n in zip(total_correct, total_num)]
        print('test acc:', acc1) 
        print(len(acc1))
        #print(f[0])

#下面是换过的
      
        total_correct2 = [0] * aa
        total_num2 = [0] * aa

        for x, label in dataset1:
            x, label = x.to(device), label.to(device)
            #f, l, u, e, y = model3(x)
            features2 = [(f, l, u, e, y)]
            #print(f[0])
            for i in range(aa):
                f, l, u, e, y = model1(x, *features2[-1])
                features2.append((f, l, u, e, y))
                
                logit = model2(y)
                pred = logit.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct2[i] += correct
                total_num2[i] += x.size(0)
                
        acc2 = [c / n for c, n in zip(total_correct2, total_num2)]
        print('test acc:', acc1,acc2) 
        print(len(acc2))



if __name__ == '__main__':
    main()




