import pandas as pd
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

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

# Definitions for the different parts of the 25 dimension embdedding space 
Frequency_area_dict = { 'V.Low' : (1,4) , 'Low' : (5,9) , 'Medium' : (10,14) , 'High' : (15,19) , 'V.High' : (20,23) }

# The below two functions 'create_token' and 'gen_random_seq' are core to creating both the harder and easier task datasets

def create_token(area):
    '''
    This function looks up frequency range, generates a Gaussian curve across 25-dim space, generates Gaussian noise which is a random 25-dim vector with fixed small mean and standard deviation. The signal and noise are then added to produce final output token. =
    
    Inputs: 
    - area (string) - one of 'V.Low' , 'Low' , 'Medium' , 'High' or 'V.High'
    Outputs: 
    - token - a 25 dimension embedding vector (numpy array)
    '''
    
    freq_range = Frequency_area_dict[area]
    # integer x for a 25-D vector
    x = np.arange(0, 25)
    
    # Generate a guassian with mean in  the specified range
    random_float = np.random.uniform(freq_range[0], freq_range[1])
    mean, std_dev = random_float, 2
    gaussian = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    #Generate a gaussian noise same num_samples as the gaussian above
    mean = 0.0125  # Mean of the Gaussian distribution
    std_dev = 0.008  # Standard deviation of the Gaussian distribution
    gaussian_noise = np.random.normal(mean, std_dev, gaussian.shape[0])

    token = gaussian + gaussian_noise
    return token   


def gen_random_seq( seq_length = 16, num_1 = -1,  keys_list = ['V.Low', 'Low', 'Medium', 'High', 'V.High']):
    '''
    Inputs: 
    - seq_length (integer) - length of sequence to be generated 
    - num_1 (int) - if set to default value of -1 the sequence starts at a random position else the generated sequence can be set to continue from a specific section
    - keys_list (list) - list of section names for embedding space
    
    Outputs:
    - generated_keys (list) - A sequence of length seq_length consisting of keys from keys_list using random walk logic. Random walk logic means that each subsequent token can only be 1 section away at most from previous token.
    '''
    
    generated_keys = []
    
    # Generate a random number and key for the sequence
    if num_1 == -1 :
        num_1 = np.random.randint(0, 5)
        generated_keys.append(keys_list[num_1])
    
    else:
        #seq length needs to be increased if were not generating the first token above
        seq_length+= 1

    
    for i in range(int(seq_length) - 1):
        #This line defines the strength of the frequency jump from token to token .. it is -1, 0 or 1 on the band level
        num = min(np.random.randint(num_1 - 1, num_1 + 2), 4)
        num = max(num, 0)
        num_1 = num # set num_1 for next token to be generated off of it
        next_key = keys_list[num_1]
        generated_keys.append(next_key)

    return generated_keys

#The below two functions gen_class_1_sequence and gen_class_2_sequence are used specifically to create the Easier Task Dataset #______________________________________________________________________________________________

def gen_class_1_sequence():
    # 5a tokens - 3a tokens - 5a tokens - 3b tokens

    """Generates a 16-token sequence for Class 1 ("Easy Task").
    This function creates a sequence with a specific, repeating pattern that defines Class 1. The structure is composed of an initial 5-token random sequence, a 3-token random sequence, a repetition of the initial 5 tokens, and a final 3-token random sequence. 
    The pattern is: 5a - 3a - 5a (repeated) - 3b.

    Returns:
        np.ndarray: A NumPy array representing the Class 1 sequence, with a shape of (25, 16).
    """
    
    keys_list = ['V.Low', 'Low', 'Medium', 'High', 'V.High']
    
    # 1. generate 5 random tokens 
    sequence_freq_words = gen_random_seq(5)
    first_5_tokens = np.stack([create_token(val) for val in sequence_freq_words], axis = 1 )

    #2. Generate 3 tokens x 2 following the 5 tokens.  
    word = sequence_freq_words[-1]
    num = keys_list.index(word)

    three_words = gen_random_seq(3, num_1 = num)
    first_3_tokens = np.stack([create_token(val) for val in three_words], axis = 1 )

    three_words_2 = gen_random_seq(3, num_1 = num)
    second_3_tokens = np.stack([create_token(val) for val in three_words_2], axis = 1 )

    #3. combine sets of tokens so you have 5 tokens - 3a tokens - 5 tokens - 3b tokens
    generated_token_sequence = np.hstack([first_5_tokens, first_3_tokens, first_5_tokens,second_3_tokens ])

    return generated_token_sequence


def gen_class_2_sequence():
    """Generates a 16-token sequence for Class 1 ("Easy Task").
    This function creates a sequence with a specific, repeating pattern that defines Class 2. The structure is composed of an initial 4-token
    random sequence, a 5-token random sequence, a repetition of the initial 4 tokens, and a final 3-token random sequence.
    
    The pattern is: 4a - 5 - 4a (repeated) - 3.

    Returns:
        np.ndarray: A NumPy array representing the Class 1 sequence, 
                    with a shape of (25, 16).
    """
    keys_list = ['V.Low', 'Low', 'Medium', 'High', 'V.High']
    
    # 1. generate 5 random tokens 
    sequence_freq_words = gen_random_seq(4)
    first_4_tokens = np.stack([create_token(val) for val in sequence_freq_words], axis = 1 )

    #2. Generate 5 tokens and 3 tokens following the 4 tokens.  
    word = sequence_freq_words[-1]
    num = keys_list.index(word)

    five_words = gen_random_seq(5, num_1 = num)
    first_five_tokens = np.stack([create_token(val) for val in five_words], axis = 1 )

    three_words = gen_random_seq(3, num_1 = num)
    first_3_tokens = np.stack([create_token(val) for val in three_words], axis = 1 )

    #3. combine sets of tokens so you have 4 atokens - 5 tokens - 4a tokens - 3 tokens
    generated_token_sequence = np.hstack([first_4_tokens, first_five_tokens, first_4_tokens,first_3_tokens])

    return generated_token_sequence

# End of Easier Task Dataset Functions _________________________________________________________________________________

# The below functions gen_5_3_5_sequence, gen_4_5_4_sequence and generate_full_token_sequence are used to create the Harder Task Dataset _______________________________________________________________________________________________


def gen_5_3_5_sequence():
    """Generates the core 13-token pattern for Class 1. This pattern consists of a sequence of 5 random tokens, followed by a
    sequence of 3 random tokens, and finally a repetition of the first 5 tokens. 
    The result is a '5-3-5_repeated' structure that defines Class 1.

    Outputs:
        np.ndarray: An array representing the Class 1 pattern, with shape (25, 13).
    """
    # 5a tokens - 3a tokens - 5a tokens
    keys_list = ['V.Low', 'Low', 'Medium', 'High', 'V.High']
    
    #1. generate 5 random tokens 
    sequence_freq_words = gen_random_seq(5)
    first_5_tokens = np.stack([create_token(val) for val in sequence_freq_words], axis = 1 )

    #2. Generate 3 tokens the 5 tokens.  
    word = sequence_freq_words[-1]
    num = keys_list.index(word)
    three_words = gen_random_seq(3, num_1 = num)
    first_3_tokens = np.stack([create_token(val) for val in three_words], axis = 1 )
  
    #3. combine sets of tokens so you have 5 tokens - 3a tokens - 5 tokens - 3b tokens
    generated_token_sequence = np.hstack([first_5_tokens, first_3_tokens, first_5_tokens])

    return generated_token_sequence


def gen_4_5_4_sequence():
    """ Generates the core 13-token pattern for Class 2.
      This pattern consists of a sequence of 4 random tokens, followed by a sequence of 5 random tokens, and finally
      repetition of the first 4 tokens. The result is a '4-5-4_repeated' structure that defines Class 2.
      
      Returns:
      np.ndarray: An array representing the Class 2 pattern, with shape (25, 13).
      """
    # 4a tokens - 5 tokens - 4a tokens
    keys_list = ['V.Low', 'Low', 'Medium', 'High', 'V.High']
    
    # 1. generate 4 random tokens 
    sequence_freq_words = gen_random_seq(4)
    first_4_tokens = np.stack([create_token(val) for val in sequence_freq_words], axis = 1 )

    #2. Generate 5 tokens following the 4 tokens.  
    word = sequence_freq_words[-1]
    num = keys_list.index(word)

    five_words = gen_random_seq(5, num_1 = num)
    first_five_tokens = np.stack([create_token(val) for val in five_words], axis = 1 )
 

    #3. combine sets of tokens so you have 5 tokens - 3a tokens - 5 tokens - 3b tokens
    generated_token_sequence = np.hstack([first_4_tokens, first_five_tokens, first_4_tokens])
    
    return generated_token_sequence


def generate_full_token_sequence(token_sequence):
    """Pads a core token sequence to a fixed length of 16. 
    This function takes a core pattern of tokens (e.g., a 13-token sequence) and adds a total of 3 random 'padding' tokens. 
    These padding tokens are randomly distributed before and after the core sequence to create a final sequence of exactly 16 tokens. 
    This is used to create the "Hard Binary Task" where the core pattern's position varies.

    Inputs:
        token_sequence (np.ndarray): The core sequence of tokens to be padded, typically with shape (25, 13).

    Outputs:
        np.ndarray: The full token sequence with padding, with shape (25, 16).
    
    """

    #There will be one,two or three tokens before the sequence
    num_pre_tokens = np.random.randint(0, 4)
    num_post_tokens = 3 - num_pre_tokens 
    
    if num_pre_tokens == 0 :
        # 0 - input_sequence - 3
        post_token_seq = np.stack([create_token(val) for val in gen_random_seq(seq_length = num_post_tokens)], axis = 1 )  
        full_token_sequence = np.hstack([token_sequence, post_token_seq])
        
    elif num_post_tokens == 0 : 
        # 3 - input_sequence - 0
        pre_token_seq = np.stack([create_token(val) for val in gen_random_seq(seq_length = num_pre_tokens)], axis = 1 )
        full_token_sequence = np.hstack([pre_token_seq, token_sequence])
        
    else:
        # 1 or 2 - input_sequence - 1 or 2
        pre_token_seq = np.stack([create_token(val) for val in gen_random_seq(seq_length = num_pre_tokens)], axis = 1 )
        post_token_seq = np.stack([create_token(val) for val in gen_random_seq(seq_length = num_post_tokens)], axis = 1 )
        full_token_sequence = np.hstack([pre_token_seq, token_sequence, post_token_seq])

    return full_token_sequence 