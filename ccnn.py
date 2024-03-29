import  torch
from    torch import nn
from    torch.nn import functional as F
import numpy as np


class INIT(nn.Module):
    def __init__(self):
        super(INIT,self).__init__()

        self.conv1_unit=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv2_unit=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1,bias=False)   
       
    def forward(self,x):
        batchsz=x.size(0)
        f = torch.zeros(batchsz,1,28,28)
        l = torch.zeros(batchsz,1,28,28)
        e = torch.zeros(batchsz,1,28,28)
        y = torch.zeros(batchsz,1,28,28)
        u = torch.zeros(batchsz,1,28,28) 
        f=f.cuda()
        l=l.cuda()
        e=e.cuda()
        y=y.cuda()
        u=u.cuda()
        
        return f,l,e,u,y

class CONV(nn.Module):
    def __init__(self):
        super(CONV,self).__init__()

        self.conv1_unit=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1,bias=False)
        self.conv2_unit=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1,bias=False)   

    def forward(self , x,f,l,e,u,y):
        
        #1
        f = np.exp(-0.1)*f + x + self.conv1_unit(y)
        l = np.exp(-1)*l + self.conv2_unit(y)    
        u = f+f*l*0.5
        e = np.exp(-1)*e + 0.1*y
        y = torch.sigmoid(u-e)

        return f,l,e,u,y
   
class FC(nn.Module):
    def __init__(self):
        super(FC,self).__init__()
        self.fc_unit = nn.Linear(28*28,10)

    def forward(self,y):
        y = y.view(-1,28*28) 
        logits = self.fc_unit(y)

        return logits
       


def main():
    net1=INIT()
    net2=CONV()
    net3=FC()

    tmp = torch.randn(56, 1,28,28)
    f,l,u,e,out = net1(tmp)
    f,l,u,e,out = net2(tmp,f,l,e,u,out)
    out = net3(out)
    print('CCNN out:', out.shape) 

if __name__ == '__main__':
    main()

