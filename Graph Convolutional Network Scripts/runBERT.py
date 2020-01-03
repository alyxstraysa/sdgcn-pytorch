import numpy as np
import tensorflow as tf
import os
import time
import datetime
import data_helpers
from sklearn import metrics
import pickle

def load_data_and_labels(positive_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(positive_data_file, "r").readlines())
    examples = [s.strip() for s in examples]
    # find the input examples
    input = []
    target = []
    for index,i in enumerate(examples):
        if index%3 == 0:
            i_target =examples[index + 1].strip()
            i = i.replace("$T$", i_target)
            input.append(i)
            target.append(i_target)
    x_text = input
    # Generate labels
    label=[]
    for index,i in enumerate(examples):
        if index%3 == 2:
            if i[0:1]=='1':
                label.append([1,0,0])
            if i[0:1]=='0':
                label.append([0,1,0])
            if i[0:1]=='-':
                label.append([0,0,1])
    y = np.array(label)
    return [x_text,target, y]
    
def preprocess():
    print("Loading Data")

    #len(train_x_str) = 3608, len(train_target_str) = 3608, train_y.shape = (3608,3)
    train_x_str, train_target_str, train_y = load_data_and_labels("data_res/bert_embedding/Restaurants_Train.txt")
    #len(test_x_str) = 1120, len(test_target_str) = 1120, test_y.shape = (1120, 3)
    test_x_str, test_target_str, test_y = load_data_and_labels("data_res/bert_embedding/Restaurants_Test.txt")

    #word embedding ---> x[324,1413,1,41,43,0,0,0]  y[0,1]
    #word_id_mapping,such as apple--->23 ,w2v  23---->[vector]
    #load the word2vec pretrained file
    #word_id_mapping, w2v = load_w2v("data_res/glove_embedding/glove.42B.300d.txt", 300)

    #print (np.shape(w2v)) = (1917497, 300)
    #print(word_dict['UNK'], len(w2v)) 1917496 1917497

    #save the dictionary (word_id_mapping) and np array (w2v) as pickl file
    # f = open("word_id_mapping.pkl","wb")
    # pickle.dump(word_id_mapping,f)
    # f.close()

    # np.savetxt('w2v', w2v)
    
    #load the pickle file
    #load the w2v embeddings
    
    with open('word_id_mapping.pkl', 'rb') as pickle_file:
        word_id_mapping = pickle.load(pickle_file)

    w2v = np.loadtxt('w2v')


    #max_document_length = max([len(x.split(" ")) for x in (train_x_str + test_x_str)])
    max_document_length = 80

    max_target_length = max([len(x.split(" ")) for x in (train_target_str + test_target_str)])
    #max_target_length = 23

    #The targets  ---->[[[141,23,45],[23,45,1,2],[2]], ...]
    #The number of targets ----> [3, ...]
    train_targets_str,train_targets_num = load_targets("data_res/bert_embedding/Restaurants_Train.txt")
    #train_targets_str = ['place', 'corner booth table', 'privacy'] (one line example) len = 3608
    #len(train_targets_num) = 3608 array([1, 1, 3, ... 6, 6, 6])
    test_targets_str, test_targets_num = load_targets("data_res/bert_embedding/Restaurants_Test.txt")
    #len(test_targets_str) = 1120

    max_target_num = max([len(x) for x in (train_targets_str + test_targets_str)])
    #max_target_num = 13

    # sentence ---> word_id
    train_x, train_x_len = word2id(train_x_str,word_id_mapping,max_document_length)
    #train_x is a (3608, 79) array (padding sentences with zero to get 79 columns) (pad to max document length)
    #train_x_len = 3608, array([ 9, 30, 31, ..., 31, 31, 31])

    test_x, test_x_len = word2id(test_x_str,word_id_mapping,max_document_length)
    #test_x = (1120, 79) array

    # target ---> word_id
    train_target, train_target_len = word2id(train_target_str,word_id_mapping,max_target_length)
    test_target, test_target_len = word2id(test_target_str,word_id_mapping,max_target_length)
    #train_target.shape = (3608, 23) (pad to max target)
    #test_target.shape(1120, 23)

    # targets ---> word_id
    train_targets, train_targets_len = word2id_2(train_targets_str,word_id_mapping,max_target_length,max_target_num)
    test_targets, test_targets_len = word2id_2(test_targets_str,word_id_mapping,max_target_length,max_target_num)
    #train_targets.shape = (3608, 13, 23), train_targets_len.shape (3608, 13)
    #test_targets.shape = (1120, 13, 23), test_targets_len.shape (1120, 13)

    #which one targets in all targets
    train_target_whichone = get_whichtarget(train_targets_num, max_target_num) #(3608, 13)
    test_target_whichone = get_whichtarget(test_targets_num, max_target_num) #(1120, 13)

    # target position
    train_target_position  = get_position("data_res/bert_embedding/Restaurants_Train.txt",max_document_length) #(3608, 79)
    test_target_position  = get_position("data_res/bert_embedding/Restaurants_Test.txt",max_document_length) # (1120, 79)

    train_x = np.load("data_res/bert_embedding/Res_Train_Embedding.npy", allow_pickle=True)                   #([3608,80,768])
    train_target = np.load("data_res/bert_embedding/Res_Train_target_Embedding.npy", allow_pickle=True)       #([3608,23,768])
    train_targets = np.load("data_res/bert_embedding/Res_Train_targets_Embedding.npy", allow_pickle=True)     #([3608,13,23,768])
    test_x = np.load("data_res/bert_embedding/Res_Test_Embedding.npy", allow_pickle=True)                     #([1120,80,768])
    test_target = np.load("data_res/bert_embedding/Res_Test_target_Embedding.npy", allow_pickle=True)         #([1120,23,768])
    test_targets = np.load("data_res/bert_embedding/Res_Test_targets_Embedding.npy", allow_pickle=True)      #([1120,13,23,768])

    train_targets_position  = get_position_2(train_target_position,train_targets_num,max_target_num)
    #train_targets_position.shape = (3608, 13, 79)
    test_targets_position  = get_position_2(test_target_position,test_targets_num,max_target_num)
    #test_targets_position.shape = (1120, 13, 79)


    #Relation Matrix
    #use test_target to create the relation
    train_relation_self,train_relation_cross = get_relation(train_targets_num, max_target_num, "global")
    #train_relation_self.shape and train_relation_cross = (3608, 13, 13)
    test_relation_self, test_relation_cross = get_relation(test_targets_num, max_target_num, "global")
    #train_relation_self.shape and train_relation_cross = (1120, 13, 13)

    Train = {'x':train_x,                       # int32(3608, 80, 768)       train sentences input embeddingID
                'T':train_target,                  # int32(3608, 23, 768)       train target input embeddingID
                'Ts':train_targets,                # int32(3608, 13, 23, 768)   train targets input embeddingID
                'x_len':train_x_len,               # int32(3608,)          train sentences input len
                'T_len':train_target_len,          # int32(3608,)          train target len
                'Ts_len': train_targets_len,       # int32(3608, 13)       train targets len
                'T_W': train_target_whichone,      # int32(3608, 13)       the ith number of all the targets
                'T_P':train_target_position,       # float32(3608, 80)
                'Ts_P': train_targets_position,    # float32(3608,13, 80)
                'R_Self': train_relation_self,     # int32(3608, 13, 13)
                'R_Cross': train_relation_cross,   # int32(3608, 13, 13)
                'y': train_y,  # int32(3608, 3)
            }

    Test = { 'x':test_x,
                'T':test_target,
                'Ts':test_targets,
                'x_len':test_x_len,
                'T_len':test_target_len,
                'Ts_len': test_targets_len,
                'T_W': test_target_whichone,
                'T_P': test_target_position,
                'Ts_P': test_targets_position,
                'R_Self': test_relation_self,
                'R_Cross': test_relation_cross,
                'y': test_y,
            }

    print("Vocabulary Size: {:d}".format(len(word_id_mapping)))
    print("Train/Test split: {:d}/{:d}".format(len(train_y), len(test_y)))

    return Train, Test, w2v

sequence_length = Train['x'].shape[1]
target_sequence_length = Train['T'].shape[1]
targets_num_max = Train['Ts'].shape[1]
num_classes = Train['y'].shape[1]
word_embedding_dim = 768
l2_reg_lambda = 0.01

model = GCN_Bert(sequence_length, target_sequence_length, targets_num_max, num_classes, word_embedding_dim, 
Train['x'], Train['T'], Train['Ts'], l2_reg_lambda)