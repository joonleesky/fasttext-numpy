import numpy as np

class FastText(object):
    def __init__(self, n_dict, hidden_dim = 10, n_classes = 4,
                 epoch = 5, lr = 1e-3, lr_decay = 0.8, initialization = 'Xavier'):
        
        self.input_dim   = n_dict
        self.hidden_dim  = hidden_dim
        self.n_classes   = n_classes
        
        self.lr = lr
        self.lr_decay = lr_decay
        self.epoch = epoch
        
        self.params = {}
        
        if initialization == 'Xavier':
            self.params['W1'] = np.random.randn(self.input_dim, hidden_dim) * np.sqrt((self.input_dim))
            self.params['W2'] = np.random.randn(hidden_dim, n_classes) * np.sqrt((hidden_dim))
        
        elif initialization == 'He':
            self.params['W1'] = np.random.randn(self.input_dim, hidden_dim) * np.sqrt(2.0 /(self.input_dim))
            self.params['W2'] = np.random.randn(hidden_dim, n_classes) * np.sqrt(2.0/(hidden_dim))
        
        
    def loss(self, X, y =None):        
        #Get loss and gradients
        #if y is None, only return prediction
        
        loss = 0
        pred = 0
        
        W1 = self.params['W1']
        W2 = self.params['W2']
        
        embed  = np.mean(W1[X],0)
        score = np.dot(embed, W2)
        
        pred = np.argmax(score)
        
        #for prediction
        if y is None:
            return pred
        
        #for update
        shifted_logits = score - np.max(score) #computation trick for softmax
        Z = np.sum(np.exp(shifted_logits))
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        
        loss = - log_probs[y]
        
        dscore = probs
        dscore[y] -= 1
        
        dW2    = np.dot(embed.reshape(self.hidden_dim,1), dscore.reshape(1, self.n_classes))
        dembed = np.dot(dscore.reshape(1,self.n_classes), W2.reshape(self.n_classes, self.hidden_dim))
        
        return loss, pred, dW2, dembed
    
    def evaluate(self,X_val,y_val):
        N = len(X_val)
        
        preds = []
        for idx in range(N):
            pred = self.loss(X_val[idx])
            preds.append(pred)
        val_acc = np.mean(np.array(preds) == y_val)
        
        return val_acc
        
    
    def train(self, X_train, y_train, X_val, y_val, verbose = 2):        
        #how many verbose per epochs
        N = len(X_train)
        
        for e in range(self.epoch):
            print('[epoch: %d]'%(e + 1))
            avg_loss  = []
            preds     = []
            start_idx = 0
            
            for idx in range(len(X_train)):
                loss, pred, dW2, dembed = self.loss(X_train[idx], y_train[idx])
                avg_loss.append(loss)
                preds.append(pred)
        
                self.params['W2'] -= self.lr * dW2
                self.params['W1'][X_train[idx]] -= self.lr * dembed
            
                if (idx + 1) % (N // verbose) == 0:
                    train_acc = np.mean(np.array(preds) == y_train[start_idx:idx+1])
                    val_acc   = self.evaluate(X_val, y_val)
                    
                    print('[idx: %6d]' %(idx), end = ' ')
                    print('loss:%.4f'%(np.mean(avg_loss)), end= ' ')
                    print('train_acc:%.4f'%(train_acc), end = ' ')
                    print('val_acc:%.4f'%(val_acc))
                    
                    start_idx = idx + 1
                    preds    = []
                    avg_loss = []
            self.lr *= self.lr_decay
