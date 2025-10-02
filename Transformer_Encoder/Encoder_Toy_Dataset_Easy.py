import os 
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib

from IPython.utils import io
import time
import sys

from scipy.signal import welch
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.signal.windows import hann

#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load dataset Easy
easy_data_path = '../Dataset/Easy_task_toy_data_binary.pkl'
processed_data_dict = joblib.load(easy_data_path)

processed_data_dict.keys()

# Process the data with the rotation
for key in ['train_data', 'val_data', 'test_data']:
    data = processed_data_dict[key]
    data_processed = [ np.flipud(np.rot90(val)) for val in data]
    processed_data_dict[key] = data_processed


# Encoder model class____________________________________________________________

class ScaledPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length, scaling_factor):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.scaling_factor = scaling_factor

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        
        # Adjusting PE so that for odd d_model it works as expected
        PE = PE[:,:self.d_model]

        # Add Scaling to the embedding dimension
        scaling_stacked = np.stack([np.linspace(self.scaling_factor , self.scaling_factor, num = int(self.d_model) , endpoint = True) for i in range(int(self.max_sequence_length))], axis = 0)
        scaling_tensor = torch.from_numpy(scaling_stacked)
        
        # Assert scaling tensor is same shape as positional encoding 
        assert scaling_tensor.shape == PE.shape, "Positional encoding and scaling tensor do not have the same shape"

        # Scale the positional encoding 
        PE = torch.mul(scaling_tensor, PE)
                
        return PE.float() #make sure the data_type is correct



class EncoderClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, max_sequence_length, scaling_factor,  pe_type = 'scaled'):
        super(EncoderClassifier, self).__init__()
        
        if pe_type == 'scaled' :
            self.position_encoder = ScaledPositionalEncoding(embed_dim, max_sequence_length , scaling_factor)  
        else:
            self.position_encoder = PositionalEncoding(embed_dim, max_sequence_length)  
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True )
        self.linear = nn.Linear(embed_dim, 1)
        self.dropout_pos_encoder = nn.Dropout(0.01)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        pos = self.position_encoder().to(x.device)
        x = (x + pos)
        x = self.encoder_layer(x)
        x = self.dropout(x)
        x = x.max(dim=1)[0]
        out = self.linear(x)
        
        return out 


# Create the training loop
# Set X_train, y_train, X_val and y_val
X_train, y_train = processed_data_dict['train_data'] , processed_data_dict['train_label']
X_val, y_val = processed_data_dict['val_data'], processed_data_dict['val_label']

#Creating train and test data loaders
train_data = [ (torch.from_numpy(spectrogram).float(), val) for spectrogram, val in zip(X_train, y_train) ] 
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

val_data = [ (torch.from_numpy(spectrogram).float(), val) for spectrogram, val in zip(X_val, y_val) ] 
val_loader = DataLoader(val_data , batch_size=64, shuffle=False)   

# Execute the training loop 
t1 = time.time()

# No seed necessary (though could be used for model only) and no fold necessary as train-validation split are already set
# print(device) - to check that device is actually cuda


for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
    
    for pe_scaling_factor in [0 , 0.01 , 0.1 , 0.2, 0.5 , 1 , 2, 10]:
        
        t3 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        criterion = nn.BCEWithLogitsLoss()

        model = EncoderClassifier(embed_dim = 25 , num_heads = 5, max_sequence_length = 16, scaling_factor = pe_scaling_factor)
        model_name = 'EncoderClassifier(embed_dim = 25 , num_heads = 5, max_sequence_length = 16)'
        
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate )

        #EPOCHS DEFINED HERE _______________________________________________________________________________________
        
        epochs = 100
        
        train_losses = []
        val_losses = []
        test_losses = []
        
        train_correct = []
        val_correct = []
        test_correct = []
        
        
        for i in range(epochs):
                if i % 5 == 0:
                    print(str(i))
                else:
                    pass
                    
                trn_corr = 0
                val_corr = 0
                tst_corr = 0
                 
                
                trn_loss = 0
                val_loss = 0
                tst_loss = 0
                
                model.train()
                # Run the training batches
                for b, (X_train_batch, y_train_batch) in enumerate(train_loader):
                    b+=1
            
                    #Move train data to the GPU
                    X_train_batch = X_train_batch.to(device)
                    y_train_batch = y_train_batch.to(device)
                    
                    # Apply the model
                    y_pred = model(X_train_batch)  # we don't flatten X-train here
                    loss = criterion(y_pred, y_train_batch.unsqueeze(1).float())
             
                    # Tally the number of correct predictions
                    predicted = torch.round(F.sigmoid(y_pred.detach() ) )
                    predicted = predicted.reshape(y_train_batch.shape)
                    
                    batch_corr = (predicted == y_train_batch).sum()
                    trn_corr += batch_corr
                    trn_loss += loss
                    
                    # Update parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                train_losses.append(trn_loss)
                train_correct.append(trn_corr)
            
                # Run the validation batches
                # Some of the variables in this loop have the same name as the variables in the above loop... be aware of that plz!
                model.eval()
                with torch.no_grad():
                    for b, (X_val_batch, y_val_batch) in enumerate(val_loader):
                        b+=1
                        
                        #Move train data to the GPU
                        X_val_batch = X_val_batch.to(device)
                        y_val_batch = y_val_batch.to(device)
            
                        # Apply the model
                        y_val = model(X_val_batch)
            
                        # Tally the number of correct predictions
                        predicted = torch.round(F.sigmoid(y_val.detach() ) )
                        predicted = predicted.reshape(y_val_batch.shape)
                        
                        batch_corr = (predicted == y_val_batch).sum()
                        val_corr += batch_corr
            
                        
                        loss = criterion(y_val, y_val_batch.unsqueeze(1).float())
                        val_loss += loss 
                       
                val_losses.append(val_loss)
                val_correct.append(val_corr)
            
        # Text for figure name and caption 
        #     
        model_name
        pe_scaling_factor = str(pe_scaling_factor)
        
            
        caption = model_name + '\n lr: ' + str(learning_rate) + '    pe_scaling: ' + pe_scaling_factor
        
        fig_name = 'dp_0.1_One_Encoder_5_heads_easy_dataset_lr_' + str(learning_rate) + '_constant_pe_scaling_' + str(pe_scaling_factor) 
        
        plt.plot([(val.cpu() / len(X_train) ) for val in train_correct], label='training set accuracy')
        plt.plot([(val.cpu()/len(X_val) ) for val in val_correct], label='validation set accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs') 
        plt.grid()
        plt.legend()
        
        # Add a styled caption
        plt.figtext(0.5, -0.06, caption , wrap=True, horizontalalignment='center', fontsize=8, fontstyle='italic', color='gray')
        plt.savefig('../Results/' + fig_name + '.png' , dpi = 100, bbox_inches='tight')
        plt.close()
        t4 = time.time()
        print(str(t4-t3))

t2 = time.time()
t2-t1