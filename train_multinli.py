
# coding: utf-8

# In[7]:


import json
import pandas as pd

import time
import argparse
import sys, os

import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer, dotdict
from models import NLINet
from nltk.tokenize import TreebankWordTokenizer
GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"


# In[8]:


def loadDataset(filename, size=-1):
    label_category = {
        'neutral': 0,
        'entailment': 1,
        'contradiction': 2
    }
    dataset = []
    sentence1 = []
    sentence2 = []
    labels = []
    with open(filename, 'r') as f:
        i = 0
        not_found = 0
        for line in f:
            row = json.loads(line, 'utf-8')
            if size == -1 or i < size:
                dataset.append(row)
                label = row['gold_label'].strip()
                if label in label_category:
                    sentence1.append( row['sentence1'].strip() )
                    sentence2.append( row['sentence2'].strip() )

                    labels.append( label_category[label] )
                    i += 1
                else:
                    not_found += 1
            else:
                break;
        if not_found > 0:
            print('Label not recognized %d' % not_found)
                
    return (dataset, sentence1, sentence2, labels)


# In[3]:


(train_dataset, train_sentence1, train_sentence2, train_labels) = loadDataset('./multinli_all/multinli_0.9_train.jsonl')
(dev_matched_dataset, dev_matched_sentence1, dev_matched_sentence2, dev_matched_labels) = loadDataset('./multinli_all/multinli_0.9_dev_matched.jsonl')
(dev_mismatched_dataset, dev_mismatched_sentence1, dev_mismatched_sentence2, dev_mismatched_labels) = loadDataset('./multinli_all/multinli_0.9_dev_mismatched.jsonl')
(test_matched_dataset, test_matched_sentence1, test_matched_sentence2, test_matched_labels) = loadDataset('./multinli_all/multinli_0.9_test_matched_unlabeled.jsonl')
(test_mismatched_dataset, test_mismatched_sentence1, test_mismatched_sentence2, test_mismatched_labels) = loadDataset('./multinli_all/multinli_0.9_test_mismatched_unlabeled.jsonl')


# In[4]:


train_set = pd.DataFrame(train_dataset)
dev_matched_set = pd.DataFrame(dev_matched_dataset)
dev_mismatched_set = pd.DataFrame(dev_mismatched_dataset)
test_matched_set = pd.DataFrame(test_matched_dataset)
test_mismatched_set = pd.DataFrame(test_mismatched_dataset)


# In[5]:


dev_matched_set = dev_matched_set[dev_matched_set['gold_label']!='-']
dev_mismatched_set = dev_mismatched_set[dev_mismatched_set['gold_label']!='-']


# In[9]:


train_sent1 = train_set['sentence1'].apply(TreebankWordTokenizer().tokenize)
train_sent2 = train_set['sentence2'].apply(TreebankWordTokenizer().tokenize)
dev_m_sent1 = dev_matched_set['sentence1'].apply(TreebankWordTokenizer().tokenize)
dev_m_sent2 = dev_matched_set['sentence2'].apply(TreebankWordTokenizer().tokenize)
dev_mis_sent1 = dev_mismatched_set['sentence1'].apply(TreebankWordTokenizer().tokenize)
dev_mis_sent2 = dev_mismatched_set['sentence2'].apply(TreebankWordTokenizer().tokenize)
test_m_sent1 = test_matched_set['sentence1'].apply(TreebankWordTokenizer().tokenize)
test_m_sent2 = test_matched_set['sentence2'].apply(TreebankWordTokenizer().tokenize)
test_mis_sent1 = test_mismatched_set['sentence1'].apply(TreebankWordTokenizer().tokenize)
test_mis_sent2 = test_mismatched_set['sentence2'].apply(TreebankWordTokenizer().tokenize)


# In[10]:


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


# In[11]:


def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))
    return word_vec


# In[12]:


def build_vocab(sentences, glove_path):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


# In[13]:


train1 = list(train_sent1)+list(train_sent2)
dev1 = list(dev_m_sent1)+list(dev_m_sent2)+list(dev_mis_sent1)+list(dev_mis_sent2)
test1 = list(test_m_sent1)+list(test_m_sent2)+list(test_mis_sent1)+list(test_mis_sent2)


# In[14]:


word_vec = build_vocab(train1+dev1+test1, GLOVE_PATH)


# In[15]:


train_set = {}
dev_matched_set = {}
dev_mismatched_set = {}
test_matched_set = {}
test_mismatched_set = {}


# In[16]:


train_set['sentence1']  =list(train_sent1   )            
train_set['sentence2']  =list(  train_sent2  )
dev_matched_set['sentence1']  =list(  dev_m_sent1  )
dev_matched_set['sentence2']  =list(  dev_m_sent2  )
dev_mismatched_set['sentence1'] =list(  dev_mis_sent1) 
dev_mismatched_set['sentence2'] =list(  dev_mis_sent2 )
test_matched_set['sentence1'] =list(  test_m_sent1 )
test_matched_set['sentence2'] =list(  test_m_sent2 )
test_mismatched_set['sentence1'] =list(  test_mis_sent1) 
test_mismatched_set['sentence2']=list(  test_mis_sent2)


# In[17]:


word_vec = build_vocab(train_set['sentence1'] +train_set['sentence2'] +dev_matched_set['sentence1'] + dev_matched_set['sentence2'] +dev_mismatched_set['sentence1'] +dev_mismatched_set['sentence2'] + test_matched_set['sentence1'] + test_matched_set['sentence2'] +test_mismatched_set['sentence1'] +test_mismatched_set['sentence2'], GLOVE_PATH )


# In[18]:


for split in ['sentence1', 'sentence2']:
    for data_type in ['train_set', 'dev_matched_set', 'dev_mismatched_set', 'test_matched_set','test_mismatched_set']:
        eval(data_type)[split] = np.array([['<s>'] + [word for word in sent if word in word_vec] +                                          ['</s>'] for sent in eval(data_type)[split]])        



# In[19]:


train_set['label'] = np.array(train_labels)
dev_matched_set['label'] = np.array(dev_matched_labels)
dev_mismatched_set['label'] = np.array(dev_mismatched_labels)


# In[20]:



GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/MultiNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir3/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')


# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

#model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")


params, _ = parser.parse_known_args(" ".split())


# In[21]:


np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)


# In[22]:


params.word_emb_dim = 300
"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder', 'InnerAttentionMILAEncoder',                 'InnerAttentionYANGEncoder', 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + str(encoder_types)
nli_net = NLINet(config_nli_model)
print(nli_net)


# In[23]:


# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
nli_net.cuda()
loss_fn.cuda()
#src_embeddings.cuda()


    
"""
TRAIN
"""
#src_embeddings.volatile = True
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None
#index_pad =word2id['<p>']


# In[24]:


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs   = []
    logs        = []
    words_count = 0
    
    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train_set['sentence1']))

    s1 = train_set['sentence1'][permutation]
    s2 = train_set['sentence2'][permutation]
    target = train_set['label'][permutation]
    

    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1                                    and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size
#         print(s1_batch, s1_len) 
        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])
        
        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data[0])
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0
        
        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)
        
        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update
        
        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append( '{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                    stidx, round(np.mean(all_costs),2), int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                    int(words_count * 1.0 / (time.time() - last_time)), round(100.*correct/(stidx+k), 2) ))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'.format(epoch, train_acc))
    return train_acc


# In[25]:


def evaluate(epoch, eval_type='valid', matched = True, final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop
    
    if eval_type == 'valid' and matched==True:
        print('\nVALIDATION, matched : Epoch {0}'.format(epoch))
    else:
        print('\nVALIDATION, mismatched : Epoch {0}'.format(epoch))
        
        
    if matched:
        data_m_set = dev_matched_set
    else:
        data_m_set = dev_mismatched_set
    
    
    s1    = data_m_set['sentence1']    
    s2    = data_m_set['sentence2']    
    target = data_m_set['label'] 


    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size
            
            
        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        
        
    # save model
    eval_acc  = round(100 * correct / len(s1),2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} : {2}'.format(epoch, eval_type, eval_acc))
    
    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net, os.path.join(params.outputdir, params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'.format(params.lrshrink, optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


# In[ ]:


epoch = 1
while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, eval_type='valid', matched = True)
    eval_acc = evaluate(epoch, eval_type='valid', matched = False)
    epoch+=1





