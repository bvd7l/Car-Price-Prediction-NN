import numpy as np 
from sklearn.model_selection import train_test_split

class NeuralNetwork: 

    def __init__(self, in_sz, lr=0.001, drop_rate=.2):
        self.lr = lr 
        self.drop_rate = drop_rate 
        
        self.ws = [] 
        self.bs = [] 
        
        self.Zs = {} 
        self.dropout_msks = {} 


        for i in range(len(in_sz) - 1): 
            kaim_he_factor = np.sqrt(2/in_sz[i])
            w = np.random.randn(in_sz[i],in_sz[i+1]) * kaim_he_factor 
            b = np.zeros((1, in_sz[i+1]))
            self.ws.append(w) 
            self.bs.append(b)  

    def relu(self,x): 
        return np.maximum(0,x) 
    
    def relu_dx(self, x): 
        return (x > 0).astype(float)

    def dropout(self, x, train=True): 
        msk = np.random.rand(*x.shape) > self.drop_rate 
        msk_scaled = x * msk / (1- self.drop_rate)
        return msk_scaled if train else x 
    
    def forward(self, x, train=True): 
        self.Zs = {} 
        curr_in = x 
        
        self.Zs['a0'] = x 
    
        for i in range(len(self.ws) - 1): 
            z = np.dot(curr_in, self.ws[i]) + self.bs[i] 
            self.Zs[f'z{i+1}'] = z 
    
            a = self.relu(z) 
            self.Zs[f'a{i+1}'] = a 
    
            if train and i < len(self.ws) - 2: 
                a = self.dropout(a, train) 
                self.Zs[f'droput_msk{i+1}'] = self.dropout_msks.get(f'dropout_msk{i+1}', None) 
            
            curr_in = a 
    
        # Move this outside the for loop - THIS IS THE FIX
        last_idx = len(self.ws) - 1
        z_out = np.dot(curr_in, self.ws[last_idx]) + self.bs[last_idx] 
        self.Zs[f'z{len(self.ws)-1}'] = z_out 
    
        return z_out
        
    def cost_fn(self, ys, ypred): 
            return np.mean( (ypred - ys) **2 )
        
    def backward(self, ys, ypred): 
        m = ys.shape[0] 
        grads_w = [] 
        grads_b = [] 

        dz = ypred - ys 
        dz = dz * (2/m) 

        for i in range(len(self.ws)-1, -1, -1): 
            a_prev = self.Zs[f'a{i}'] if i > 0 else self.Zs['a0']
            dw = np.dot(a_prev.T, dz) 
            db = np.sum(dz, axis=0, keepdims=True) 

            grads_w.insert(0, dw) 
            grads_b.insert(0,db) 

            if i > 0: 
                da = np.dot(dz, self.ws[i].T) 
                z_prev = self.Zs[f'z{i}'] 
                dz = da * self.relu_dx(z_prev) 

        return grads_w, grads_b 
    
    def update(self, grads): 
        for i in range(len(self.ws)): 
            self.ws[i] -= self.lr * grads[0][i] 
            self.bs[i] -= self.lr * grads[1][i] 

    def batch(self, x_batch, y_batch): 
        ypred = self.forward(x_batch, train=True) 

        loss = self.cost_fn(y_batch, ypred) 

        grads = list(self.backward(y_batch, ypred))

        self.update(grads)

        return loss 
    
    def split_train_validation(self, x_train, y_train, dev_split=.2):
        n = x_train.shape[0] 
        val_sz = int(n*dev_split) 
        indices = np.random.permutation(n) 
        train = indices[val_sz:] 
        val = indices[:val_sz] 

        x_train_split = x_train[train] 
        y_train_split = y_train[train] 
        x_val = x_train[val]
        y_val = y_train[val] 

        return x_train_split, y_train_split, x_val, y_val 

    def predict(self, x_in, train=False): 
        return self.forward(x_in, False) 

    def fit(self, x_train, y_train, epoc=100, batch_size=32, dev_split=.2, verbose=1): 
        x_train_split, y_train_split, x_val, y_val = self.split_train_validation(x_train, y_train, dev_split=dev_split)  # FIX: use dev_split parameter
        n = x_train_split.shape[0] 
        records = {
            'loss': [],
            'val_loss': [],
            'mae': [],
            'val_mae': []
        }
        
        for epoch in range(epoc):
            # Shuffle data each epoch
            indices = np.random.permutation(n)
            x_shuffled = x_train_split[indices]
            y_shuffled = y_train_split[indices]
            
            epoch_losses = []
            
            for i in range(0, n, batch_size): 
                x_batch = x_shuffled[i:i+batch_size] 
                y_batch = y_shuffled[i:i+batch_size] 
                loss = self.batch(x_batch, y_batch) 
                epoch_losses.append(loss)
            
            # Record metrics every epoch
            train_pred = self.predict(x_train_split)
            train_loss = self.cost_fn(train_pred, y_train_split)
            train_mae = np.mean(np.abs(train_pred - y_train_split)) 
            
            val_pred = self.predict(x_val) 
            val_loss = self.cost_fn(val_pred, y_val) 
            val_mae = np.mean(np.abs(val_pred - y_val)) 
            
            records['loss'].append(train_loss) 
            records['mae'].append(train_mae) 
            records['val_loss'].append(val_loss) 
            records['val_mae'].append(val_mae) 
            
            if verbose and epoch % 10 == 0:  
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        return records