# -*- coding: utf-8 -*-
"""
Tesing of Pytorch Version.
This is a temporary script file.
"""

import torch
#import cifar
import torchvision
import torchvision.transforms as transforms
import timeit
import math
import numpy as np
#from scipy import linalg
import gc
#from PIL import Image, ImageOps, ImageEnhance

def compute_mean_var(data,list_mean,list_var):
    bn_numpy = data.cpu().numpy()
    bn_mean = np.mean(bn_numpy,(0,2,3))
    bn_var = np.var(bn_numpy,(0,2,3)) 
    list_mean.append(torch.from_numpy(bn_mean))
    list_var.append(torch.from_numpy(bn_var))
    return None

def compute_mean_var_fc(data,list_mean,list_var):
    bn_numpy = data.cpu().numpy()
    bn_mean = np.mean(bn_numpy,0)
    bn_var = np.var(bn_numpy,0) 
    list_mean.append(torch.from_numpy(bn_mean))
    list_var.append(torch.from_numpy(bn_var))
    return None

def reduce_mean_var(list_mean,list_var, batchSize):
    length = len(list_mean)
    print(length)
    running_mean = list_mean[0] / length
    running_var = list_var[0] / length
    for index, item in enumerate(list_mean,1):
        running_mean += item / length
    for idex, item in enumerate(list_var,1):
        running_var += item / length
    running_var *= batchSize / (batchSize - 1)
    return running_mean, running_var
        
        

def net_eval(net):
    net = net.train()
    for name, module in net.named_modules():
        if 'drop' in name:
            """
            set dropout layer in eval() mode.
            """
            """
            while the batchNorm layer is difficult to be handled for accuracy.
            """
            module.eval()
    return net

def net_freeze(net):
    net = net.train()
    for name, module in net.named_modules():
        if 'grelu' in name:
            """
            set dropout layer in eval() mode.
            """
            """
            while the batchNorm layer is difficult to be handled for accuracy.
            """
            module.eval()
    return net

def test(net, criterion, testLoader):
    start = timeit.default_timer()          
    net.cuda()
    correct = 0
    total = 0
    test_loss = 0
    minibatch = 0
    #class_correct = list(0. for i in range(10))
    #class_total = list(0. for i in range(10))
    #net.eval()
    for data in testLoader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        minibatch += 1
        test_loss += criterion(outputs, labels).data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.data.size(0)
        correct += (predicted == labels.data).sum()
        #c = (predicted == labels.data).squeeze()
        """
        for i in range(100):
            label = labels.data[i].cpu()
            class_correct[label] += c[i].cpu()
            class_total[label] += 1
        """
    #print(total)
    #print(minibatch*100)
    test_loss /= minibatch
    test_rate = 100. * correct / total
    print('Accuracy of the network on the %f test images: %f %%' % (
        total, test_rate))
    print('Average loss of test dataset: %f' %(test_loss))
    end = timeit.default_timer()
    print('consumes %f seconds' %(end-start))
    return test_rate, test_loss


from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as u
import torch.nn.functional as F
from GreluModule import GreluModule as grelu

"""
data loader
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
start = timeit.default_timer()
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
print(trainset.train_data.shape)
#print(trainset.train_labels.shape)
#print(testset.test_data.shape)
print(trainset[0][0].size())
#trainset.train_data, testset.test_data = data_init(trainset.train_data, testset.test_data)
end = timeit.default_timer()
print('data preprocessing: %f' %(end-start))
#trainset.transform = transform
#testset.transform = transform
#print(torch.mean(trainset.train_data))
#print(np.max(trainset.train_labels))
#print(np.min(trainset.train_labels))
#print(trainset.train_data[0].shape)
#print(trainset.train_labels[0])
#print(trainset[0][0].size())
print(testset[0][0].size())
#print(torch.mean(testset.test_data))
print(torch.max(testset.test_labels))
print(torch.min(testset.test_labels))
#print(trainset.train)
#print(trainset.train_list)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)


"""
class net definition.
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        """
        set a variable 
        if 0: representing the network in training stage, 
        elif 1: in final setting stage (computing the batchNorm running_mean and running_var)
        else 2: final inference stage with computed running_mean and running_var from final tunning stage.
        """
        #self.tiny = 1e-10
        self.infer = 0
        self.dropData = nn.Dropout(p=0.1)
        self.dropConv1 = nn.Dropout(p=0.5)
        self.dropConp1 = nn.Dropout(p=0.5)
        self.dropConv2 = nn.Dropout(p=0.3)
        self.dropConp2 = nn.Dropout(p=0.5)
        self.dropConv3 = nn.Dropout(p=0.3)
        self.dropConp3 = nn.Dropout(p=0.5)
        self.dropFC = nn.Dropout(p=0.0)
        self.conv1 = u.weight_norm(nn.Conv2d(1, 96, 3, padding=1, bias=False),name='weight')
        init.kaiming_normal(self.conv1.weight,a=0.0,mode='fan_in')
        self.conp1 = u.weight_norm(nn.Conv2d(96, 96, 1, bias=False),name='weight')
        init.kaiming_normal(self.conp1.weight,a=0.00,mode='fan_in')
        #self.conp11 = u.weight_norm(nn.Conv2d(128 * 2, 128, 1, bias=False),name='weight')
        #init.kaiming_normal(self.conp1.weight,a=0.00,mode='fan_in')
        self.bn11 = nn.BatchNorm2d(96)
        self.bn11_mean = list()
        self.bn11_var = list()
        #self.bn13 = nn.BatchNorm2d(128)
        self.bn12 = nn.BatchNorm2d(96)
        self.bn12_mean = list()
        self.bn12_var = list()
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        """
        only apply maxPool on the first pooling layer.
        """
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        #self.pool2 = nn.AvgPool2d(3, 2, 1)
        self.avg1 = nn.AvgPool2d(4, 4)
        self.avg2 = nn.AvgPool2d(2, 2)
        self.conv2 = u.weight_norm(nn.Conv2d(96, 96, 3, padding=1, bias=False),name='weight')
        init.kaiming_normal(self.conv2.weight,a=0.0,mode='fan_in')
        self.conp2 = u.weight_norm(nn.Conv2d(96 + 96, 96, 1, bias=False),name='weight')
        init.kaiming_normal(self.conp2.weight,a=0.00,mode='fan_in')
        self.bn21 = nn.BatchNorm2d(96)
        self.bn21_mean = list()
        self.bn21_var = list()
        self.bn22 = nn.BatchNorm2d(96)
        self.bn22_mean = list()
        self.bn22_var = list()
        self.conv3 = u.weight_norm(nn.Conv2d(96, 96, 3, padding=1, bias=False),name='weight')
        init.kaiming_normal(self.conv3.weight,a=0.0,mode='fan_in')
        self.conp3 = u.weight_norm(nn.Conv2d(96 + 96, 96, 1, bias=False),name='weight')
        init.kaiming_normal(self.conp3.weight,a=0.0,mode='fan_in')
        self.bn31 = nn.BatchNorm2d(96)
        self.bn32 = nn.BatchNorm2d(96)
        self.bn31_mean = list()
        self.bn31_var = list()
        self.bn32_mean = list()
        self.bn32_var = list()
        #self.avg3 = nn.AvgPool2d(4,4)
        self.fc = u.weight_norm(nn.Conv2d( ( 96 + 96 + 96 ), 10, 1, bias=False),name='weight')
        init.kaiming_normal(self.fc.weight,a=0.00,mode='fan_in')
        self.bn = nn.BatchNorm2d(10)
        self.bn_mean = list()
        self.bn_var = list()
        self.gap = nn.AvgPool2d(7,7)
        self.grelu1 = grelu(2)
        self.grelu2 = grelu(2)
        self.grelu3 = grelu(2)
        self.grelu4 = grelu(2)
        self.grelu5 = grelu(2)
        self.grelu6 = grelu(2)
    def reduce_bn_statistics(self,batchSize):
        if self.infer == 2:
            self.bn11.running_mean,self.bn11.running_var = reduce_mean_var(self.bn11_mean,self.bn11_var, batchSize)
            self.bn12.running_mean,self.bn12.running_var = reduce_mean_var(self.bn12_mean,self.bn12_var, batchSize)
            self.bn21.running_mean,self.bn21.running_var = reduce_mean_var(self.bn21_mean,self.bn21_var, batchSize)
            self.bn22.running_mean,self.bn22.running_var = reduce_mean_var(self.bn22_mean,self.bn22_var, batchSize)
            self.bn31.running_mean,self.bn31.running_var = reduce_mean_var(self.bn31_mean,self.bn31_var, batchSize)
            self.bn32.running_mean,self.bn32.running_var = reduce_mean_var(self.bn32_mean,self.bn32_var, batchSize)
            self.bn.running_mean,self.bn.running_var = reduce_mean_var(self.bn_mean,self.bn_var, batchSize)
        return None
    def clear_bn_statistics(self):
        if self.infer == 0:
            self.bn11_mean, self.bn11_var = list(), list()
            self.bn12_mean, self.bn12_var = list(), list()
            self.bn21_mean, self.bn21_var = list(), list()
            self.bn22_mean, self.bn22_var = list(), list()
            self.bn31_mean, self.bn31_var = list(), list()
            self.bn32_mean, self.bn32_var = list(), list()
            self.bn_mean, self.bn_var = list(), list()
    def forward(self, x):
        x = self.dropData(x)
        """
        block 1
        """
        x1 = self.conv1(x)
        if self.infer == 1:
            compute_mean_var(x1.data,self.bn11_mean,self.bn11_var)
        x1 = self.grelu1(self.bn11(x1))
        x1 = self.dropConv1(x1)
        x1 = self.conp1(x1)
        if self.infer == 1:
            compute_mean_var(x1.data,self.bn12_mean,self.bn12_var)
        x1 = self.dropConp1(F.relu(self.bn12(x1)))
        x1a = self.avg1(x1)
        """
        pooling
        """
        x2 = self.pool1(x1)
        x2_cat = x2
        """
        block 2
        """
        x2 = self.conv2(x2)
        
        if self.infer == 1:
            compute_mean_var(x2.data,self.bn21_mean,self.bn21_var)
        
        x2 = self.grelu3(self.bn21(x2))
        """
        Applies Dense Operation to sufficiently retain information flows from lower layers.
        It seems better to cat feature maps then 
        dropout to
        avoid overfitting.
        """
        x2 = torch.cat((x2_cat,x2),1)
        x2 = self.conp2(self.dropConv2(x2))
        
        if self.infer == 1:
            compute_mean_var(x2.data,self.bn22_mean,self.bn22_var)
        
        x2 = self.bn22(x2)
        x2 = self.dropConp2(F.relu(x2))
        """
        pooling
        """
        x2a = self.avg2(x2)
        x3 = self.pool2(x2)
        x3_cat = x3
        """
        block 3.
        """
        x3 = self.conv3(x3)
        
        if self.infer == 1:
            compute_mean_var(x3.data,self.bn31_mean,self.bn31_var)
        
        x3 = self.grelu5(self.bn31(x3))
        x3 = torch.cat((x3_cat,x3),1)
        x3 = self.dropConv3(x3)
        x3 = self.conp3(x3)
        
        if self.infer == 1:
            compute_mean_var(x3.data,self.bn32_mean,self.bn32_var)
        
        x3 = F.relu(self.bn32(x3))
        x3 = self.dropConp3(x3)
        """
        Now catenate the features together before 
        pooling and generate NumOfClass feature maps 
        for global average pooling.
        """
        #x3 = self.avg3(x3)
        """
        Final layer
        """
        #x1a = x1a.view(-1, 128 * 2 * 2)
        #x2a = x2a.view(-1, 160 * 2 * 2)
        #x3 = x3.view(-1, 192 * 2 *2)
        x4 = torch.cat((x1a, x2a, x3), 1)
        #x4 = self.dropFC(x4)
        x5 = self.fc(x4)
        
        if self.infer == 1:
            compute_mean_var(x5.data,self.bn_mean,self.bn_var)
        """
        Global average pooling.
        """
        x6 = self.gap(self.bn(x5))
        """
        Setting to be 2-D (batchSize * NumOfClasses)
        """
        x6 = x6.view(-1,10)
        #print('after batchnorm before grelu: '+str(x6[0:1,:]))
        #print('mean of each feature: '+ str(torch.mean(x6,dim=0)))
        #print("mean of all: " + str(torch.mean(x6)))
        #x7 = self.lr(x6)
        #return torch.log(F.softmax(x6) + self.tiny)
        return x6
        #return x6


    
if __name__ == '__main__':
    gc.enable()
    """
    normalize it into a standard nrom N(0,1) distribution.
    """
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    import matplotlib.pyplot as plt
    #import numpy as np
    
    # functions to show an image
    def imshow(img):
        #img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    # get some random training images
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()
    
    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(10)))
    """
    Create an instance.
    """
    net = Net()
    net.train()
    net.cuda()
    best_test_rate = 0
    best_test_loss = 10
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = lr.StepLR(optimizer, step_size=100, gamma=0.1)
    start = timeit.default_timer()
    for epoch in range(400):  # loop over the dataset multiple times
    
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            #inputs, labels = inputs.float(),labels.float()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #for name,para in net.named_parameters():
            #    if 'lr' in name:
            #       print(name)
            #       print(para)
            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:    # print loss every 10000 examples, i.e., 5 printouts per epoch.
            #if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                end = timeit.default_timer()
                print('time usage: %f' %(end-start))
                start = end
                running_loss = 0.0
        """
        
        """
        #test_start = timeit.default_timer()
        """
        running average of batchNorm parameters.
        """
        #test_rate_eval = test(net.eval(), testloader)
        """
        train mode of test accuracy, more convincing for me.
        Need to write a small program to get the correct mean and variance values for testing.
        Work to do.
        """
        
        test_rate_train, test_loss = test(net_eval(net), criterion, testloader)
        if best_test_rate < test_rate_train:
            best_test_rate = test_rate_train
            best_net = net.state_dict()
            torch.save(best_net,'mnist_net')
        if best_test_loss > test_loss:
            """
            the model with least loss will also be considered in future.
            """
            best_test_loss = test_loss
            best_net_loss = net.state_dict()
            torch.save(best_net_loss,'mnist_net_loss')
        test_end = timeit.default_timer()
        print('best rate until recently: %f' %(best_test_rate))
        #print('test time usage: %f' %(test_end-test_start))
        start = test_end
    
    print('Finished Training')  
    
    test(net_eval(net), criterion, testloader)                            
    net.cuda()
    correct = 0
    total = 0
    test_loss = 0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))
    #net.load_state_dict(best_net)
    bestNet = Net()
    bestNet.load_state_dict(torch.load('mnist_net'))
    bestNet = net_eval(bestNet)
    bestNet = bestNet.cuda()
    minibatch = 0
    for data in testloader:
        images, labels = data
        images, labels = Variable(images), Variable(labels)
        images, labels = images.cuda(), labels.cuda()
        outputs = bestNet(images)
        test_loss += criterion(outputs, labels).data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.data.size(0)
        minibatch += 1
        correct += (predicted == labels.data).sum()
        c = (predicted == labels.data).squeeze()
        for i in range(100):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    test_loss /= minibatch
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    print('Average loss of test dataset: %f' %(test_loss))
    for i in range(100):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
    """
    Fun: checking pytorch structure.
    """
    gc.collect()
    for name,para in net.named_parameters():
        if 'lr' in name:
            print(name)
            print(para)
