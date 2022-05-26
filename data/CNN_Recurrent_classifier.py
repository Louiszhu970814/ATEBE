#!/usr/bin/env python
# coding: utf-8

# # CNNs for Heart Rate Estimation and Human Activity Recognition in Wrist Worn Sensing Applications

# This is code for reproducing the CNNR HRE results shown in the paper presented at the WristSense workshop as part of PerCom 2020.
# 
# This repository will be broken down as shown in the Figure 1 below.

# ![](./Block_Diagram_LS.png)
# 
# Figure 1. *Block diagram of our processing approach*

# ## Data Collection

# The data was collected by [D. Jarchi and A. Casson (2017)](https://www.mdpi.com/2306-5729/2/1/1) and downloaded from [PhysioNet](https://physionet.org/content/wrist/1.0.0/).
# 
# After running `Download_Data.ipynb` the data will be downloaded and stored in the directory `'/CNNs_HAR_and_HR/Data/wrist/'`

# ### Using Google Colaboratory - (Recommended Working Environment)

# You can run this notebook on Colab using the following cell to mount your drive and install some dependencies

# You may need to install some of these packages below

# In[8]:


import os

import wfdb
from wfdb import processing

import matplotlib.pyplot as plt
import numpy as np
import json as js

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

torch.manual_seed(0)

# Change cwd
path = 'data/WristSense/CNNs_HAR_and_HR'
os.chdir(path)


# ## Load Data
# 
# This step is done in by selecting each exercise at a time. We can begin with the 'walk' exercise.
# 
# By changing the word below betwwen 'high', 'low', 'run', 'walk' we can pre-process our data.

# In[ ]:


def load_data(fileDir, exercise):
    word = exercise
    file_path_list = []
    valid_file_extensions = [".dat"]
    valid_file_extensions = [item.lower() for item in valid_file_extensions]


    for file in os.listdir(fileDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_file_extensions:
            continue
        file_path_list.append(os.path.join(fileDir, file))

    PPG = []
    ECG = []
    for path in file_path_list:
        base=os.path.basename(path)
        base = os.path.splitext(base)[0]
        if word in base:
            #sample = wfdb.rdsamp('Data/wrist/%s'%(base))
            sample = wfdb.rdsamp(fileDir+'/%s'%(base))
            PPG.append(sample[0][:,1])
            ECG.append(sample[0][:,0])

    PPG = np.asarray(PPG)
    ECG = np.asarray(ECG)

    return PPG, ECG


# ## Segment, Normalise and Downsample Data
# 
# 
# ```slidingWindow()``` returns a generator that iterates through the input sequence.

# In[ ]:


def slidingWindow(sequence,winSize=2048,step=256):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
 
    # Verify the inputs
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
 
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence)-winSize)//step)+1
 
    # Do the work
    for i in range(0,numOfChunks*step,step):
        yield sequence[i:i+winSize]


# ### PPG and ECG
# 
# Here the PPG signal is:
# 1. Segmented using the ```slidingWindow``` function
# 2. Normalised between 0 and 1 (for each sample) using ```wfdb.processing.normalise_bound```
# 3. Cleaned (Removed any rows with NaNs from segmentation)
# 4. Downsampled using the downsampling factor ```ds_factor```.
# 
# The returned signal **p** is ready to be used in the experiments
# 
# The ECG signal is segmented here using the ```slidingWindow``` function and then cleaned as above.

# In[ ]:


def preprocess_PPG_ECG(PPG, ECG, downsample=True, ds_factor=25):
    prev_p = []
    e = []

    for i in range(len(PPG)):
        ppg = slidingWindow(PPG[i], winSize = 2048)
        for sig in ppg:
            nrm_sig = processing.normalize_bound(sig, lb=0, ub=1) # normalise signal
            prev_p.append(nrm_sig)

    for i in range(len(ECG)):
        ecg = slidingWindow(ECG[i], winSize = 2048)
        for sig in ecg:
            e.append(sig)

    prev_p = np.asarray(prev_p, dtype=np.float32)  #290 and 781
    e = np.asarray(e, dtype=np.float32)


    p = prev_p[~np.isnan(prev_p).any(axis=1)] # Remove rows with NaN
    e = e[~np.isnan(prev_p).any(axis=1)] # Remove rows corresponding to PPG NaN

    # Don't need to downsample ECG
    if downsample == True:
        p = p[:,::ds_factor]
    print(p.shape)
    return p,e


# In[ ]:


import numpy as np
a = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
a=np.array(a)
a = a[:,::2]
print(a)


# ### Load Ground Truth of ECG

# Get QRS of ECG signal using ```wfdb.processing.XQRS()```

# In[ ]:


def gt_ECG(ecg):
    
    y = []
    for i in range(len(ecg)):
        sig = ecg[i, :]
        xqrs = processing.XQRS(sig=sig, fs=256.0)
        xqrs.detect()

        HR = processing.compute_hr(len(sig), xqrs.qrs_inds, 256.0)
        HR = HR[np.logical_not(np.isnan(HR))] # Remove any NaN in HR Array
        AvgHR = (np.mean(HR))

        y.append(AvgHR)

    y = np.asarray(y, dtype=np.float32)
    y = np.around(y)

    max_y = max(y)
    y = y/max(y)
    print(y.shape)
    
    return y, max_y


# ## RCNN Model

# In[ ]:


class RCNN(nn.Module):
    def __init__(self, input_size, batch_size, n_features, 
                 cv1_k, cv1_s, cv2_k, cv2_s,
                 cv3_k, cv3_s, cv4_k, cv4_s):
        super(RCNN, self).__init__()
    
        self.input_size = input_size
    
        self.cv1_k = cv1_k
        self.cv1_s = cv1_s
        self.cv1_out = int(((self.input_size - self.cv1_k)/self.cv1_s) + 1)

        self.cv2_k = cv2_k
        self.cv2_s = cv2_s
        self.cv2_out = int(((self.cv1_out - self.cv2_k)/self.cv2_s) + 1)

        self.cv3_k = cv3_k
        self.cv3_s = cv3_s
        self.cv3_out = int(((self.cv2_out - self.cv3_k)/self.cv3_s) + 1)

        self.cv4_k = cv4_k
        self.cv4_s = cv4_s
        self.cv4_out = int(((self.cv3_out - self.cv4_k)/self.cv4_s) + 1)
    
        self.layer_1 = nn.Sequential(
          nn.Conv1d(in_channels=1, out_channels=5, kernel_size=(self.cv1_k), stride=(self.cv1_s)),
          nn.BatchNorm1d(num_features=3),
          nn.ReLU(inplace=True),
          nn.AvgPool1d(kernel_size=1)
        )

        self.layer_2 = nn.Sequential(
          nn.Conv1d(in_channels=3, out_channels=4, kernel_size=(self.cv2_k), stride=(self.cv2_s)),
          nn.BatchNorm1d(num_features=4),
          nn.ReLU(inplace=True),
          nn.AvgPool1d(kernel_size=1)
        )
        
        # self.layer_hidden = nn.Sequential(
        #   nn.Conv1d(in_channels=4, out_channels=5, kernel_size=(self.cv2_k), stride=(self.cv2_s)),
        #   nn.BatchNorm1d(num_features=5),
        #   nn.ReLU(inplace=True),
        #   nn.AvgPool1d(kernel_size=1)
        # )

        self.layer_3 = nn.Sequential(
          nn.Conv1d(in_channels=5, out_channels=8, kernel_size=(self.cv3_k), stride=(self.cv3_s)),
          nn.BatchNorm1d(num_features=8),
          nn.ReLU(inplace=True)
        )

        self.layer_4 = nn.Sequential(
          nn.Conv1d(in_channels=8, out_channels=10, kernel_size=(self.cv4_k), stride=(self.cv4_s)),
          nn.BatchNorm1d(num_features=10),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5, inplace=False)
        )

        self.layer_5 = nn.Sequential(
          nn.Linear(self.cv4_out*10, 20), # FC Layer
          nn.Linear(20, 1) # Regression
        )
        
    def forward(self, x):
        x = self.layer_1(x) 
        x = self.layer_2(x)
        x = self.layer_hidden(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = x.view(x.size(0), -1)
        x = self.layer_5(x)
    
        return x


# In[ ]:


def call_RCNN(seq_len, batch_size, cv1_k = 1, cv1_s = 1, cv2_k = 1, cv2_s = 1,
              cv3_k = 1, cv3_s = 1, cv4_k = 3, cv4_s = 3):
    
    rcnn = RCNN(input_size = seq_len, batch_size = batch_size, n_features = 1, 
                   cv1_k = cv1_k, cv1_s = cv1_s,
                   cv2_k = cv2_k, cv2_s = cv2_s,
                   cv3_k = cv3_k, cv3_s = cv3_s,
                   cv4_k = cv4_k, cv4_s = cv4_s)

    rcnn = rcnn.cuda()
    
    return rcnn


# ### Optimiser and Loss functions and Train/Test Split

# In[ ]:


def dataloaders(rcnn, ppg, target, batch_size):
    optimizer = torch.optim.SGD(rcnn.parameters(), lr=0.001)
    loss_func = nn.MSELoss() # mean squared loss for regression
    loss_func = loss_func.cuda() # may need to check if cuda() available

    x = torch.from_numpy(ppg)
    y = torch.from_numpy(target)
    
    x, y = Variable(x), Variable(y) # torch trains on Variable, so convert.

    #DataLoader
    dataset_loader = DataLoader(x, batch_size=batch_size, shuffle=True)
    
    # Creat list of data and targets
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i]])

    num_batches = len(ppg)//batch_size 
    # test to be 10% of data and train to be the rest
    test_percent = int(num_batches*0.1)
    test_split = batch_size*test_percent
    train_split = (len(data) - (test_split))

    print(test_split)
    print(train_split)

    train_dataset,test_dataset= torch.utils.data.random_split(data,(train_split, test_split))

    trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
    
    return optimizer, loss_func, trainloader, testloader


# ## Train RCNN

# In[ ]:


def train(epochs, batch_size, seq_len, rcnn, trainloader, testloader, optimizer, loss_func, error_msg, exer, frequency):
    num_epochs = epochs

    total_loss = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.view(batch_size, 1, seq_len)
            labels = labels[:,None]
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = rcnn(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        
            if i % 10 == 9:    # print every 10 mini-batches
                print('Epoch:[%d / %d], Batch: [%d],  Loss: %.3f' %
                      (epoch + 1, num_epochs, i + 1, running_loss / 10))
                total_loss.append(running_loss/10)
                running_loss = 0.0
        if epoch%10==0:
            error = test_rcnn(batch_size, seq_len, rcnn, testloader, max_y)
        
            print("Error: " +str(error))

            error_msg['hr_errors'].append({
            'exercise': str(exer),
            'batch_size': str(batch_size),
            'frequency': str(frequency)+'Hz',
            'error': str(error)
            })
            
    print('Finished Training...')
    
    return error_msg


# ## Test RCNN for HR Error Rate

# In[ ]:


def heart_rate_difference(labels, predictions, max_y):
    labels.shape
    predictions.shape
    difference = []
    
    for i in range(len(labels)):
        target = labels[i].detach().cpu().numpy()
        guess = predictions[i].detach().cpu().numpy()

        target = np.around(target*max_y)
        guess = np.around(guess*max_y)
        difference.append(abs(target-guess) / target)

    d = np.asarray(difference)
    d = np.mean(d)
  
    return d


# In[ ]:


def test_rcnn(batch_size, seq_len, rcnn, testloader, max_y):
    hre = []
    for i, data in enumerate(testloader, 0): #test_Loader
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
      
        inputs = inputs.view(batch_size, 1, seq_len)
        labels = labels[:,None]
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        # forward + backward + optimize
        predictions = rcnn(inputs)    
        difference = heart_rate_difference(labels, predictions, max_y)
        hre.append(difference*100)
    return np.mean(hre)


# ## Main - function calls
# 
# When you call the RCNN function you can specify Conv-Pooling params which will affect the outcome of your heart rate error. \\
# Your choice of conv-pooling ```filter (cv_k)``` and ```stride (cv_k)``` sizes will be dependent on  ```seq_len``` that changes with you downsampling factors ```dwns_factor```.  You can set these in the ```call_RCNN()``` function\\
# 
# The results will be written to a json file in format: \\
# [```batch_size```, ```exercise```, ```downsampled frequency```, ```heart rate error```]

# ---------------
# It is recommended to leave the downsampling factor `dwns_factor` commented out in the second cell below, it was found to be computationally heavy to run through all the samples at once.
# 
# Therefore please select downsample factor param `dwns_factor` in cell immediately below to pass through the loop with one sampling frequency.

# In[ ]:


#@title Select your Downsampling Factor
# dwns_factor = "8" #@param [1, 8, 17, 25, 51]

# dwns_factor = [dwns_factor]





# Exercises in dataset
exercise = ['high', 'low', 'run', 'walk']
# Original sampling frequency
fs = 256.0

# Downsampling Factor - can be computationally heavy to run through loop
# with all dwns_factor at once, therefore select Downsample factor in cell above

dwns_factor = [fs//10.0]


epoch = 150
# File Directory for data
fileDir='./Data/wrist'

downsample=True
error_msg = {}
error_msg['hr_errors'] = []
for exer in exercise:
    for d in dwns_factor:
        # Load Data
        PPG, ECG = load_data(fileDir, exer)
            # Preprocess Data
        d = int(d)

        if d == 1:
          downsample == False
        else:
          downsample == True

        ppg, ecg = preprocess_PPG_ECG(PPG, ECG, downsample=downsample, ds_factor=d)

            # Fix batching as dataset is not balanced
            # Can choose any of the values in the comments - this will affect results

        if exer == 'high':  # 28, 30, 35, 36, 42, 45
          batch_size = 28 
        elif exer == 'low': # 34, 51
          batch_size = 34
        elif exer == 'run': # 26, 28, 52, 56
          batch_size = 28
        else:               # 37, 59
          batch_size = 37


        seq_len = len(ppg[0,:])

            # Get ECG HR Ground Truth
        ecg_groundTruth, max_y =  gt_ECG(ecg)

        # Call RCNN Model
        # You can specify Conv-Pooling params in call
        # Your choice should be dependent on  seq_len
        rcnn = call_RCNN(seq_len, batch_size)
        # Set up data into train/test splits with targets
        optimizer, loss_func, trainloader, testloader = dataloaders(rcnn, ppg, ecg_groundTruth, batch_size)
        # Train the model
        error_msg =  train(epoch, batch_size, seq_len, rcnn, trainloader, testloader, optimizer, loss_func, error_msg, exer, frequency=fs//d)
        # Test the model/estimate HRE
        # error = test_rcnn(batch_size, seq_len, trained_rcnn, testloader, max_y)
        
        # print("Error: " +str(error))

        # error_msg['hr_errors'].append({
        # 'exercise': str(exer),
        # 'batch_size': str(batch_size),
        # 'frequency': str(fs//d)+'Hz',
        # 'error': str(error)
        # })
    
        json = js.dumps(error_msg)
        f = open('./Results/'+'errors_2.json','w')
        f.write(json)
        f.close()

# json = js.dumps(error_msg)
# f = open('./Results/'+str(256.0//int(d))+str(epoch)+'Hz_errors.json','w')
# f.write(json)
# f.close()

# ## Load Back Results

# # In[ ]:


# result_file = ('./Results/'+str(256.0//int(d))+'Hz_errors.json')

# with open(result_file) as json_file:
#     data = js.load(json_file)
#     for p in data['hr_errors']:
#         print(p)

