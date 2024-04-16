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
plt.ion()
from    ccnn import INIT,CONV,FC
from visdom import Visdom

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

unloader = transforms.ToPILImage()



def main():
    batchsz=200
    mnist_train = datasets.MNIST('../data',train=True,download=True,
        transform=transforms.Compose([transforms.Resize((28,28)),
                            transforms.ToTensor()
                             ]))
    mnist_train = DataLoader(mnist_train,batch_size =batchsz,shuffle=True)

    mnist_test = datasets.MNIST('../data',train=False,download=True,
        transform=transforms.Compose([transforms.Resize((28,28)),
                            transforms.ToTensor()
                            ]))
    mnist_test = DataLoader(mnist_test,batch_size =batchsz,shuffle=True)

  
 
    x, label = iter(mnist_test).next()
    print('x:', x.shape, 'label:', label.shape)
    

    device = torch.device('cuda')
    model1 = INIT().to(device)
    model2 = CONV().to(device)
    model3 = FC().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer1 = optim.Adam(model3.parameters(), lr=1e-3)
    optimizer2 = optim.Adam(model1.parameters(), lr=1e-3)
    optimizer3 = optim.Adam(model2.parameters(), lr=1e-3)
    print(model1)
    print(model2)
    print(model3)
    viz=Visdom()
                                      
    global_step = 0
    train_losses = []
    train_accs = []


    for epoch in range(55):
       
        model3.train()
        model1.train()
        model2.train()
        total_correct = 0
        total_num = 0
        for batchidx, (x, label) in enumerate(mnist_train):

            x, label = x.to(device), label.to(device)
            f,l,u,e,y = model1(x)
            f,l,u,e,y = model2(x,f,l,u,e,y)
           
        


            logits = model3(y)
            loss = criteon(logits, label)
            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
            xx=x
            yy=y
            optimizer3.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer3.step()
            optimizer1.step()
            optimizer2.step()

        train_losses.append(loss.item())
        train_acc = total_correct / total_num
        train_accs.append(train_acc)
        

        global_step += 1
        """ viz.line([loss.item()], [global_step], win='train_loss', 
                    update='append',opts=dict(title='train_loss')) 
        viz.line([train_acc],[global_step],win='train_acc',
                    update='append',opts=dict(title='train_acc'))
        """
        #viz.images(yy.view(-1, 1, 28, 28), win='yy',opts=dict(title='yy'))
        print(epoch, 'train_loss:', loss.item(),'train_acc:',train_acc)

        #print(u[:1])
        if (epoch>46,loss.item()<0.26):
            torch.save(model1,'net1.pth')
            torch.save(model2,'net2.pth')
            torch.save(model3,'net3.pth')

    print('loss',train_losses,'acc',train_accs)

if __name__ == '__main__':
    main()
    