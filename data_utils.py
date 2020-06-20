import numpy as np
import nltk
import csv
import re

class Data(object):
    def __init__(self, 
                 train_data_source, test_data_source,
                 n_grams = 2, split_ratio = 0.9, max_length = 300):

        #n_grams : how many ngrams to use
        
        #data -- train  -- X
        #               -- y
        #     -- val    -- X
        #               -- y
        #     -- test   -- X
        #               -- y
        
        self.train_data_source = train_data_source
        self.test_data_source = test_data_source
        self.n_grams = n_grams
        self.split_ratio = split_ratio
        self.max_length = max_length
        
        self.dict  = {}
        self.inverse_dict = {}
        
        self.data = {}
        self.data['train'] = {}
        self.data['test']  = {}
        
    
    def loadData(self):
        #Load Train Data
        data, labels   = [], []
        with open(self.train_data_source, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                data.append(row[1:])
                labels.append(int(row[0]))
            
        self.data['train']['X'] = np.asarray(data)
        self.data['train']['y'] = np.asarray(labels) - 1
        
        #Load Test Data
        data, labels   = [], []
        with open(self.test_data_source, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                data.append(row[1:])
                labels.append(int(row[0]))
        
        self.data['test']['X'] = np.asarray(data)
        self.data['test']['y'] = np.asarray(labels) - 1
        
        print('train data num:',len(self.data['train']['X']))
        print('test data num:',len(self.data['test']['X']))
        
        
    def preProcess(self):
        for key in self.data:
            data = []
            for row in self.data[key]['X']:
                s = ''
                for sentence in row:
                    s += re.sub('[^a-zA-Z]', ' ', sentence) #only considering alphabets
                s = s.lower()                               #lower
                s = nltk.word_tokenize(s)
                if(len(s) == 0):
                    data.append(['unk'])
                else:
                    data.append(s)
            
            self.data[key]['X'] = data
    
    #bi-gram dictionary
    def build_dictionary(self):
        self.dict['UNK'] = 0
        for row in self.data['train']['X']:
            for idx in range(len(row)):
                key = row[idx]
                if key not in self.dict:
                    self.dict[key] = len(self.dict)
        
        if self.n_grams == 2:
            for row in self.data['train']['X']:
                for idx in range(len(row)-1):
                    key = (row[idx],row[idx+1])
                    if key not in self.dict:
                        self.dict[key] = len(self.dict)
        
        for key,value in self.dict.items():
            self.inverse_dict[value] = key   
    
    def strToIdx(self):
        for key in self.data:
            data = []
            for row in self.data[key]['X']:
                sentence = []
                for unigram in row:
                    if unigram in self.dict:
                        sentence.append(self.dict[unigram])
                    else:
                        sentence.append(self.dict['UNK'])
                    
                if self.n_grams == 2:
                    for idx in range(len(row)-1):
                        bigram = (row[idx],row[idx+1])
                        if bigram in self.dict:
                            sentence.append(self.dict[bigram])
                        else:
                            sentence.append(self.dict['UNK'])
                data.append(sentence)
            
            self.data[key]['X'] = np.asarray(data)

    def splitData(self):
        self.data['val']   = {}
        
        #shuffle index
        N = len(self.data['train']['X'])
        rand_idx = np.arange(N)
        np.random.shuffle(rand_idx)
        
        #split
        split_idx = int(N * self.split_ratio)
        data   = self.data['train']['X'][rand_idx]
        labels = self.data['train']['y'][rand_idx]
        
        self.data['train']['X'] = data[:split_idx]
        self.data['train']['y'] = labels[:split_idx]
        
        self.data['val']['X'] = data[split_idx:]
        self.data['val']['y'] = labels[split_idx:]
        
        
        
        
        
        
