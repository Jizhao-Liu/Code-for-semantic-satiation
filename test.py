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
    """  idx0 = (dataset0.targets==3) 
    dataset0.targets = dataset0.targets[idx0]
    dataset0.data = dataset0.data[idx0]
    #dataset0.train_data = dataset0.data.unsqueeze(1) """

    print(dataset0.data.size())
    dataset0 = DataLoader(dataset0,batch_size = batchsz,shuffle=True,drop_last=True)

  
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
        models = [model3] + [model1] * 100
        total_correct = [0] * 100
        total_num = [0] * 100
        
        for x, label in dataset0:
            x, label = x.to(device), label.to(device)
            f, l, u, e, y = model3(x)
            features = [(f, l, u, e, y)]
            
            global_step = 0

            for i in range(100):
                f, l, u, e, y = model1(x, *features[-1])
                features.append((f, l, u, e, y))
                
                logit = model2(y)
                pred = logit.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct[i] += correct
                total_num[i] += x.size(0)
                #print(y.size())

                #viz.images(e.view(-1, 1, 28, 28), win='e',opts=dict(title='e'))
                #viz.images(f.view(-1, 1, 28, 28), win='f',opts=dict(title='f'))
                viz.images(u.view(-1, 1, 28, 28), win='u',opts=dict(title='u'))
                #viz.images(l.view(-1, 1, 28, 28), win='l',opts=dict(title='l')) 

        acc = [c / n for c, n in zip(total_correct, total_num)]
        print('test acc:', acc) 
        print(len(acc))
        


if __name__ == '__main__':
    main()




