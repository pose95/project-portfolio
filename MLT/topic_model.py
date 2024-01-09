#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import re
import json
import codecs
import copy
import numpy as np
import datetime
from optparse import OptionParser
from sklearn.preprocessing import normalize
from scipy import sparse
from scipy.io import savemat
from spacy.lang.en import English 
import file_handling as fh

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import Counter
from itertools import chain
#from progressbar import ProgressBar
#from ipdb import launch_ipdb_on_exception
#from tqdm import tqdm
import gc
import time
ISOTIMEFORMAT='%Y-%m-%d %X'
#import heapq
#import random

#from scipy import sparse
#from scipy.io import savemat
#import pandas as pd
from subprocess import Popen, PIPE


# In[3]:


def log(logfile, text, write_to_log=True):
    if write_to_log:
        with codecs.open(logfile, 'a', encoding='utf-8') as f:
            f.write(text + '\n')


# In[4]:


def load_data(input_dir, input_prefix, log_file, vocab=None):
    print("Loading data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    X = np.array(temp, dtype='float32')  #矩阵：数据条数 * 词数    值表示出现的次数
    del temp
    temp2 = fh.load_sparse(os.path.join(input_dir, input_prefix + '_X_indices.npz')).todense()
    indices = np.array(temp2, dtype='float32') 
    del temp2
    lists_of_indices = fh.read_json(os.path.join(input_dir, input_prefix + '.indices.json'))
    index_arrays = [np.array(l, dtype='int32') for l in lists_of_indices]
    del lists_of_indices
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))  #排序后词表
    n_items, vocab_size = X.shape
#     assert vocab_size == len(vocab)
    
    label_file = os.path.join(input_dir, input_prefix + '.labels.npz')
    if os.path.exists(label_file):
        print("Loading labels")
        temp = fh.load_sparse(label_file).todense()
        labels = np.array(temp, dtype='float32')    ##数据条数 * label数20 
    else:
        print("Label file not found")
        labels = np.zeros([n_items, 1], dtype='float32') 
    assert len(labels) == n_items
    gc.collect()
    
    return X, vocab, labels,indices,index_arrays


# In[5]:


class DataIter(object):
    def __init__(self, data, indices, index_arrays, batch_size):
        self.data = data
        self.indices = indices
        self.index_arrays = index_arrays
        assert len(self.data) == len(self.indices)
        assert len(self.data) == len(self.index_arrays)
        self.batch_size = batch_size
        self.num_document = len(self.data)
        self.num_batch = self.num_document // self.batch_size
    def __iter__(self):
        self.batch_permuted_list = np.random.permutation(self.num_batch)
        self.batch_i = 0
        return self
    def __next__(self):
        if self.batch_i >= self.num_batch:
            raise StopIteration
        else:
            self.batch_index = self.batch_permuted_list[self.batch_i]
            starting_point = self.batch_index * self.batch_size
            if starting_point + self.batch_size >= self.num_document:
                end_point = self.num_document
            else:
                end_point = starting_point + self.batch_size
            batch_data = self.data[starting_point:end_point]
            batch_indices = self.indices[starting_point:end_point]
            batch_index_arrays = self.index_arrays[starting_point:end_point]
            self.batch_i += 1
        return batch_data,batch_indices,batch_index_arrays


# In[6]:


class TopicModel(nn.Module):
    def __init__(self,d_v, d_e, d_t, encoder_layers=1, generator_layers=4, 
                 encoder_shortcut=False, generator_shortcut=False,generator_transform=None ):
        
        super(TopicModel, self).__init__()
        
        self.d_v = d_v  # vocabulary size
        self.d_e = d_e  # dimensionality of encoder
        self.d_t = d_t  # number of topics
        self.encoder_layers = encoder_layers
        self.generator_layers = generator_layers

        # set various options
        self.generator_transform = generator_transform   # transform to apply after the generator
        self.encoder_shortcut = encoder_shortcut
        self.generator_shortcut = generator_shortcut
            
        self.en1_fc = nn.Linear(self.d_v, self.d_e)
        self.en2_fc = nn.Linear(self.d_e, self.d_e)
        self.en_drop = nn.Dropout(0.2)
        self.mean_fc = nn.Linear(self.d_e, self.d_t)
#         self.mean_bn = nn.BatchNorm1d(self.d_t)
        self.logvar_fc = nn.Linear(self.d_e, self.d_t)
#         self.logvar_bn = nn.BatchNorm1d(self.d_t)
        
        self.generator1 = nn.Linear(self.d_t, self.d_t)
        self.generator2 = nn.Linear(self.d_t, self.d_t)
        self.generator3 = nn.Linear(self.d_t, self.d_t)
        self.generator4 = nn.Linear(self.d_t, self.d_t)
        
        self.r_drop = nn.Dropout(0.2)
        
        self.de = nn.Linear(self.d_t, self.d_v)
#         self.de_bn = nn.BatchNorm1d(self.d_v)


    
    def encoder(self, x):
        if self.encoder_layers == 1:
            pi = F.relu(self.en1_fc(x))
            if self.encoder_shortcut:
                pi = self.en_drop(pi)
        else:
            pi = F.relu(self.en1_fc(x))
            pi = F.relu(self.en2_fc(pi))
            if self.encoder_shortcut:
                pi = self.en_drop(pi)

#         mean = self.mean_bn(self.mean_fc(pi))
#         logvar = self.logvar_bn(self.logvar_fc(pi))
        mean = self.mean_fc(pi)
        logvar = self.logvar_fc(pi)
        return mean, logvar

    def sampler(self, mean, logvar):
        eps = Variable(torch.randn(mean.size()).cuda())
        sigma = torch.exp(logvar)
        h = sigma.mul(eps).add_(mean)
        return h
      
    def generator(self, h):
        if self.generator_layers == 0:
            r = h
        elif self.generator_layers == 1:
            temp = self.generator1(h)
            if self.generator_shortcut:
                r = F.tanh(temp) + h
            else:
                r = temp
        elif self.generator_layers == 2:
            temp = F.tanh(self.generator1(h))
            temp2 = self.generator2(temp)
            if self.generator_shortcut:
                r = F.tanh(temp2) + h
            else:
                r = temp2
        else:
            temp = F.tanh(self.generator1(h))
            temp2 = F.tanh(self.generator2(temp))
            temp3 = F.tanh(self.generator3(temp2))
            temp4 = self.generator4(temp3)
            if self.generator_shortcut:
                r = F.tanh(temp4) + h
            else:
                r = temp4

        if self.generator_transform == 'tanh':
            return self.r_drop(F.tanh(r))
        elif self.generator_transform == 'softmax':
            return self.r_drop(F.softmax(r)[0])
        elif self.generator_transform == 'relu':
            return self.r_drop(F.relu(r))
        else:  
            return self.r_drop(r)
        
    def decoder(self, r):
#         p_x_given_h = F.softmax(self.de_bn(self.de(r)))
        p_x_given_h = F.softmax(self.de(r))
        return p_x_given_h    
                
    def forward(self, x):
        mean, logvar = self.encoder(x)
        h = self.sampler(mean, logvar)
        r = self.generator(h)
        p_x_given_h = self.decoder(r)
        
        return mean, logvar, p_x_given_h
    


# In[7]:


def print_topics(model, vocab, log_file, topic_num, write_to_log=True):
    vocab = {value:key for key,value in vocab.items()}
    n_topics = model.d_t
    highest_topic_list = []
    if n_topics > 1:
        log(log_file, "Topics:", write_to_log)
        weights = model.de.weight.detach().cpu().numpy()
#         mean_sparsity = 0.0
        for j in range(n_topics):
            highest_list = []
            order = list(np.argsort(weights[:, j]).tolist()) #返回的是数组值从小到大的索引值
            order.reverse()
            k = 0
            for i in order:
                if k>=topic_num:
                    break
                if i in vocab:
                    highest_list.append(vocab[i])
                    k+=1
                    
            highest = ' '.join(highest_list)
            print("%d %s" % (j, highest))
            log(log_file, "%d %s" % (j, highest), write_to_log)


# In[9]:


def get_reward_cv(model, vocab, log_file,topic_file):

    vocab = {value:key for key,value in vocab.items()}
    n_topics = model.d_t
    total_cv = 0
    topic_list = []
    if n_topics > 1:
        weights = model.de.weight.detach().cpu().numpy()
        for j in range(n_topics):
            highest_list = []
            order = list(np.argsort(weights[:, j]).tolist()) #返回的是数组值从小到大的索引值
            order.reverse()
            k = 0
            for i in order:
                if k>=5:
                    break
                if i in vocab:
                    highest_list.append(vocab[i])
                    k+=1
#             highest_list = [vocab[i] for i in order[:5]]
            topic_list.append(highest_list)
        f = open(topic_file,'w')
        for topic in topic_list:
            for word in topic:
                f.write(word+' ')
            f.write('\n')
        f.close()
        p = Popen(['java', '-jar', 'palmetto-0.1.0-jar-with-dependencies.jar', 'wikipedia_bd/','C_V',topic_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate(b"input data that is passed to subprocess' stdin")
        temp_list = output.decode("utf-8").split("\n")
        
        for i,line in enumerate(temp_list):
            if re.search(r"\t(.*)\t",line):
                co_herence_cv = re.search(r"\t(.*)\t",line)
                cv = float(co_herence_cv.group(1))
                total_cv+=cv
        cv = total_cv/(i+1)  #1*d_t
        
    return cv


# In[35]:


def evaluate(log_file,model,test_X_dataset,vocab,cuda=True):
    model.train(False)
#     test_X = normalize(np.array(test_X, dtype='float32'), axis=1)
    n_items, dv = test_X.shape
    bound = 0
    batch_index_arrays_num = 0
    batch_num = 0
    bound2 = 0
    
    for i,batch in enumerate(test_X_dataset):
        batch_data,batch_indices,batch_index_arrays = batch[0],batch[1],batch[2]
        if cuda is None:
            x = torch.from_numpy(np.array(batch_data,dtype='float32'))
            x_indices = torch.from_numpy(np.array(batch_indices,dtype='float32'))
        else:
            x = torch.from_numpy(np.array(batch_data,dtype='float32')).cuda()
            x_indices = torch.from_numpy(np.array(batch_indices,dtype='float32')).cuda()

        mean, logvar, p_x_given_h = model(x)
        loss, nll_term, KLD, penalty = loss_function(x, mean, logvar, p_x_given_h,x_indices)
        
#         index_length_list = []
#         for index_array in batch_index_arrays:
#             index_length_list.append(len(index_array))
#         index_length_array = np.array(index_length_list)
#         bound += (nll_term.detach().cpu().numpy()/index_length_array).mean()
#         batch_num+=1
        
        counts_list = []
        for i in x:
            counts_list.append(torch.sum(i).detach().cpu())
        if np.mean(counts_list) != 0:
            bound += (nll_term.detach().cpu().numpy()/ np.array(counts_list)).mean()
            batch_num += 1
        
#     print('none empty document:'+ str(batch_index_arrays_num))
#     bound = np.exp(bound/float(batch_index_arrays_num))
    bound = np.exp(bound/float(batch_num))
    print("Estimated perplexity upper bound on test set = %0.3f" % bound)
    log(log_file, "Estimated perplexity upper bound on test set = %0.3f" % bound)
    
    return bound


# In[11]:


def train(log_file,model_file,model,optimizer_tm,
          train_X_dataset, test_X_dataset,vocab,
          max_epochs,topic_num, temp_batch_num,topic_file, cuda=True):
    print("Start Tranining")
    log(log_file, "Start Tranining")
    if cuda != None:
        model.cuda()
        
#     train_X = normalize(np.array(train_X, dtype='float32'), axis=1)
    epochs_since_improvement = 0
    min_bound = np.inf
    cv_list = []
#     self_dictionary.id2token = {k: v for v, k in self_dictionary.token2id.iteritems()}
    start =time.clock()
    for epoch_i in range(max_epochs):
        print("\nEpoch %d" % epoch_i)
        print(time.strftime( ISOTIMEFORMAT, time.localtime() ))
        log(log_file, "\nEpoch %d" % epoch_i)
        log(log_file, time.strftime( ISOTIMEFORMAT, time.localtime()))
        
        temp_batch_index = 0
        running_cost = 0
        batch_num = 0
        bound = 0
        model.train(True)
        for i,batch in enumerate(train_X_dataset):
            batch_data,batch_indices,batch_index_arrays = batch[0],batch[1],batch[2]
            if cuda is None:
                x = torch.from_numpy(np.array(batch_data,dtype='float32'))
                x_indices = torch.from_numpy(np.array(batch_indices,dtype='float32'))
            else:
                x = torch.from_numpy(np.array(batch_data,dtype='float32')).cuda()
                x_indices = torch.from_numpy(np.array(batch_indices,dtype='float32')).cuda()

            mean, logvar, p_x_given_h = model(x)
            optimizer_tm.zero_grad()            
            loss, nll_term, KLD, penalty = loss_function(x, mean, logvar, p_x_given_h, x_indices)
            running_cost += loss
            loss.mean().backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
            optimizer_tm.step()
            
        print("\nepoch batch nll_term KLD l1p loss")
        print("%d %d %0.4f %0.4f %0.4f %0.4f" % (epoch_i, i, nll_term.mean(), KLD.mean(), penalty, loss.mean()))
        log(log_file, "\nepoch batch nll_term KLD l1p loss")
        log(log_file, "%d %d %0.4f %0.4f %0.4f %0.4f" % (epoch_i, i, nll_term.mean(), KLD.mean(), penalty, loss.mean()))
        bound = evaluate(log_file,model,test_X_dataset,cuda)
            
        co = get_reward_cv(model, vocab, log_file,topic_file)
        cv_list.append(co)
        print('cv: '+ str(co))
        log(log_file, "co_herence_cv: %0.3f" % float(co))
        
        if bound < min_bound:
            print("New best dev bound = %0.3f" % bound)
            log(log_file, "New best dev bound = %0.3f" % bound)
            min_bound = bound
#             print("Saving model")
#             torch.save(model.state_dict(),model_file)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print("No improvement in %d epoch(s)" % epochs_since_improvement)
            log(log_file, "No improvement in %d batches(s)" % epochs_since_improvement)
            
#         print_topics(model, vocab, log_file, topic_num, write_to_log=True)
#         if epochs_since_improvement >= 10:
#             break 

    print("The best dev bound = %0.3f" % bound)
    log(log_file, "The best dev bound = %0.3f" % min_bound)
    log(log_file, "Final topics:")
    print_topics(model, vocab, log_file, topic_num, write_to_log=True)
    end =time.clock()
    print('Running time: %s Seconds'%(end-start))
    
    return cv_list


# ## Main

# In[31]:


def loss_function(x, mean, logvar, p_x_given_h, indices):

    KLD = -0.5 * torch.sum((1 + logvar - (mean ** 2) - torch.exp(logvar)),1)

    nll_term = -torch.sum(torch.mul(x,torch.log(p_x_given_h+1e-10)),1)
    #avitm
#     nll_term = -(x * (p_x_given_h+1e-10).log()).sum(1)
    loss = KLD+nll_term

    # add an L1 penalty to the decoder terms
#     penalty = l1_strength * (torch.sum(torch.abs(parameter)).data[0])
    penalty = 0       
    return loss,nll_term, KLD, penalty


# In[32]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[36]:


test_prefix = 'test'
batchsize=32
topic_vocabsize=2000
temp_batch_num=1000
lr=1e-4
de = 500
dt = 50
encoder_layers = 1
generator_layers = 4
min_epochs = int(200)
max_epochs = int(500)
topic_num = int(50)
l1_strength = np.array(0.0, dtype=np.float32)

input_dir = 'data/20ng/20ng_all/LR'
input_prefix = 'train'
output_prefix = 'output'
# model_file = os.path.join(input_dir, 'TopicModel_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize)+ '_encoder' + str(encoder_layers)+ '_generator' + str(generator_layers) + '.pkl')
# log_file = os.path.join(input_dir, 'TopicModel_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize)+ '_encoder' + str(encoder_layers)+ '_generator' + str(generator_layers) + '.log')
model_file = os.path.join(input_dir, 'TopicModel_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize) + '_dt' + str(dt)+ '.pkl')
log_file = os.path.join(input_dir, 'TopicModel_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize)+ '_dt' + str(dt)+ '.log')
topic_file = os.path.join(input_dir, 'TopicModel_batchsize' + str(batchsize) + '_vocab' + str(topic_vocabsize)+ '_dt' + str(dt)+ '.txt')

# load data
#矩阵：数据条数 * 词数    值表示出现的次数
#排序后词表
##每行矩阵存放该文档中所有词的index
##数据条数 * label数20 
train_X, vocab, train_labels,train_indices,train_index_arrays = load_data(input_dir, input_prefix, log_file)
# train_X = normalize(np.array(train_X, dtype='float32'), axis=1)
test_X, _, test_labels,test_indices,test_index_arrays = load_data(input_dir, test_prefix, log_file, vocab)
# test_X = normalize(np.array(test_X, dtype='float32'), axis=1)
n_items, dv = train_X.shape
n_items, dy = train_labels.shape
print(train_X.shape, train_labels.shape)


# In[ ]:


train_X_dataset = DataIter(train_X, train_indices, train_index_arrays, batchsize)
test_X_dataset = DataIter(test_X, test_indices, test_index_arrays, batchsize)

model = TopicModel(topic_vocabsize, de, dt, encoder_layers, generator_layers, encoder_shortcut=False, generator_shortcut=False,generator_transform=False)
optimizer_tm = optim.Adam(model.parameters(),lr = lr)
cv_list = train(log_file,model_file,model,optimizer_tm,train_X_dataset, test_X_dataset, vocab,
      max_epochs,topic_num, temp_batch_num, topic_file, cuda=True)
# evaluate(log_file,model,test_X_dataset,vocab,cuda=True)


# In[62]:


#np.max(np.array(cv_list))


# In[ ]:


#1114


# In[50]:


#30:chuntrain:164.64317499999999/50


# In[25]:


#30:203.57038699999998/50


# In[22]:


#50:211.253664/50


# ### plot

# In[63]:


import numpy as np
import matplotlib.pyplot as plt
y = cv_list
x = list(range(0,len(y)))
plt.figure(figsize=(8,4)) #创建绘图对象
plt.plot(x,y,"b--",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
plt.xlabel("Epoch(es)") #X轴标签
plt.ylabel("Co_herence_cv")  #Y轴标签
# plt.title("cv plot") #图标题
# plt.axis([-5,100,0,0.7])
plt.show()  #显示图


# In[ ]:




