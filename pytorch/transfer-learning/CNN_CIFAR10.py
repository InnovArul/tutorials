#!/usr/bin/env python
# coding: utf-8

# ## Training CNN on CIFAR10 Dataset [1]
# 
# ### CIFAR10 dataset
# ![CIFAR10](../pics/cifar10.png)

# ## Components of a DL Project

# ### Dataloader and Transformers
# 
# 
# To make data loading simple, we would use the torchvision package created as part of PyTorch which has data loaders for standard datasets such as ImageNet, CIFAR10, MNIST.
# 

# In[1]:


#a Tensor library with GPU support
import torch

#Datasets, Transforms and Models specific to Computer Vision
import torchvision
import torchvision.transforms as transforms

####train data
#Compose transforms (applies data transformation and augmentation) prior to feeding to training
train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(64),
    #  transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
    transforms.Resize(64),
     #transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#inbuilt dataset class for reading CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=train_transform)
print("train : " + str(len(trainset)) + ' images')

train_samples = 'all'
sampler = None

if train_samples != 'all':
    # collect the indices for each class
    indices = {}
    
    for i, (data, label) in enumerate(trainset):
        if label in indices:
            indices[label].append(i)
        else:
            indices[label] = []
            indices[label].append(i)
    
    print(len(indices[0]))
    final_indices = []
    for i in range(10):
        final_indices += indices[i][:train_samples]
    
    print(len(final_indices))
    subset_sampler = torch.utils.data.SubsetRandomSampler(indices=final_indices)
    sampler = subset_sampler

#dataloader for Batching, shuffling and loading data in parallel
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          num_workers=2, sampler=sampler)

print("train : " + str(len(trainloader) * 4) + ' images')

####test data
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=False, transform=test_transform)
print("test : " + str(len(testset)) + ' images')

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

print(("image size : ", testset[0][0].size()))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# #### Visualizing the dataset images

# In[2]:


#plotting and visualization library
import matplotlib.pyplot as plt
#Display on the notebook
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.ion() #Turn interactive mode on.

#scientific computing library for Python
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# # Options

# In[3]:


class options():
    def __init__(self):
        self.pretrained = True
        self.use_gpu = True
        self.freeze_non_fc = False

opts = options()


# # ResNet18 network
# 

# In[4]:


#a neural networks library integrated with autograd functionality
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    
    #define the learnable paramters by calling the respective modules (nn.Conv2d, nn.MaxPool2d etc.)
    def __init__(self, pretrained):
        super(Net, self).__init__()
        
        #features
        #self.features = models.alexnet(pretrained=pretrained).features
        self.model = models.resnet18(pretrained=pretrained)
        
        #classifier
        self.model.fc = nn.Sequential(nn.Dropout(0.3), 
                                        nn.Linear(512, 10))
        
#                                         nn.Dropout(0.3),
#                                         nn.ReLU(),
#                                         nn.Linear(512, 10))

    
    #defining the structure of the network
    def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.shape[0], -1)
#         x = self.classifier(x)
        x = self.model(x)
        return x

net = Net(pretrained=opts.pretrained)

#Printing the network architecture
print(net)


# In[5]:


#Printing the parameter values
params = list(net.parameters())
for name, param in net.named_parameters():
    if 'fc' not in name and opts.freeze_non_fc:
        print(name, ' frozen')
        param.requires_grad = False
    print(name, param.shape) 


# #### Forward Pass

# In[6]:


input = torch.randn(1, 3, 224, 224)
out = net(input)
print(out)


# #### Backward Pass

# In[7]:


net.zero_grad()
out.backward(torch.randn(1, 10))


# #### Loss Function
# In this example, we will use Classification Cross-Entropy loss and SGD with momentum.<br>
# Cross Entropy loss is given as:- $L=-\sum_i y_i \log(p_i)$ and $p_i=\frac{\exp^{x_i}}{\sum_k \exp^{x_k}}$
# 
# There are many other loss functions such as MSELoss, L1Loss etc. Visit [here](http://pytorch.org/docs/master/nn.html#loss-functions) for other loss functions.

# In[8]:


criterion = nn.CrossEntropyLoss()
print(criterion)


# 
# #### Stochastic Gradient Descent (SGD)
# $$w_{n+1} = w_{n} - \eta \triangle$$
# $$\triangle = 0.9\triangle + \frac{\partial L}{\partial w}$$
# 
# Although SGD is the most popular and basic optimizer that one should first try. There are many adaptive optimizers like Adagrad,Adadelta RMSProp and many more. Visit [here](http://pytorch.org/docs/master/optim.html) for other examples.

# In[9]:


#an optimization package with standard optimization methods such as SGD, RMSProp, LBFGS, Adam etc.
import torch.optim as optim
params = [param for param in net.parameters()
                 if param.requires_grad]
print(len(params))
optimizer = optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=5e-4)


# #### Training in mini-batches

# In[10]:


########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^

def train(epoch, trainloader, optimizer, criterion):
    running_loss = 0.0
    
    net.train()
    
    for i, data in enumerate(tqdm(trainloader), 0):
        # get the inputs
        inputs, labels = data
        if opts.use_gpu and torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # addup loss
        running_loss += loss.item()

    print('epoch %d training loss: %.3f' %
            (epoch + 1, running_loss / (len(trainloader))))
    return running_loss / (len(trainloader))   


# #### Forward Pass over the trained network

# In[11]:


outputs = net(images)
_, predicted = torch.max(outputs.data, 1)


imshow(torchvision.utils.make_grid(images))
print('Predicted: ', ' '.join(['%5s' % classes[predicted[j]] for j in range(4)]))


# ### Test Accuracy 

# In[12]:


########################################################################
# Let us look at how the network performs on the test dataset.

def test(testloader, model):
    running_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            # get the inputs
            inputs, labels = data
            if  opts.use_gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # addup loss
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%, loss = %f' % (
                                    100 * correct / total, running_loss / len(testloader)))
    return running_loss / len(testloader)


# ### Class-wise accuracy
# 

# In[13]:



def classwise_test(testloader, model):
########################################################################
# class-wise accuracy

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if opts.use_gpu and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


# In[14]:


import os
from tqdm import tqdm as tqdm
print('Start Training')
if not os.path.exists('./models'):
    os.mkdir('./models')

training_losses = []
testing_losses = []
num_epochs = 10

if  opts.use_gpu and torch.cuda.is_available():
    net = net.cuda()

for epoch in range(num_epochs):  # loop over the dataset multiple times
    print('epoch ', epoch + 1)
    train_loss = train(epoch, trainloader, optimizer, criterion)
    test_loss = test(testloader, net)
    classwise_test(testloader, net)
    torch.save(net.state_dict(), './models/model-'+str(epoch)+'.pth')
    
    training_losses.append(train_loss)
    testing_losses.append(test_loss)

print('Finished Training')


# In[15]:


#Plotting the training graph
plt.plot(range(len(training_losses)), training_losses, label="train")
plt.plot(range(len(testing_losses)), testing_losses, label="test")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

