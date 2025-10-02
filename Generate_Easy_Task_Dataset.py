import joblib
import numpy

# Import the functions required to generate the easy dataset
from Generate_Dataset_Functions import gen_class_1_sequence, gen_class_2_sequence


# Specify the number of samples per class in the train, validation and test 
# Default is 50,0000-10,000-10,000 for train,val,test per class 
# Default Total dataset size is 140,000 with 100,000-20,000-20,000 train-val-test split and 50/50 class balance
num_samples_train , num_samples_val , num_samples_test = 50000 , 10000 , 10000 


data_dict = {}
data_dict_keys = ['train_data', 'train_label' , 'val_data' , 'val_label' , 'test_data' , 'test_label' ]

for key in data_dict_keys:
    data_dict[key] = []

#Generate samples - per class 50,000 in train, 10,000 in validation , 10,000 in test data 
data_dict['train_data'] += [ gen_class_1_sequence() for i in range(num_samples_train) ]
data_dict['train_label'] += [ 0 for i in range(num_samples_train) ]
data_dict['train_data'] += [ gen_class_2_sequence() for i in range(num_samples_train) ]
data_dict['train_label'] += [ 1 for i in range(num_samples_train) ]

data_dict['val_data'] += [ gen_class_1_sequence() for i in range(num_samples_val) ]
data_dict['val_label'] += [ 0 for i in range(num_samples_val) ]
data_dict['val_data'] += [ gen_class_2_sequence() for i in range(num_samples_val) ]
data_dict['val_label'] += [ 1 for i in range(num_samples_val) ]

data_dict['test_data'] += [ gen_class_1_sequence() for i in range(num_samples_test) ]
data_dict['test_label'] += [ 0 for i in range(num_samples_test) ]
data_dict['test_data'] += [ gen_class_2_sequence() for i in range(num_samples_test) ]
data_dict['test_label'] += [ 1 for i in range(num_samples_test) ]


joblib.dump(data_dict, 'Dataset/Easy_task_toy_data_binary.pkl')