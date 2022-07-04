# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils import smiles2adjoin
import tensorflow as tf


# MG-BERT Dataset
# str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
#          'B': 10,'I': 11,'Si':12,'Se':13,'<unk>':14,'<mask>':15,'<global>':16}

# DrugBank Dataset
str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
         'I': 10,'Na': 11,'Fe':12,'Mg':13,'<unk>':14,'<mask>':15,'<global>':16}

num2str =  {i:j for j,i in str2num.items()}



class Graph_Bert_Dataset(object):
    def __init__(self,path,smiles_field='SMILES',addH=True):  # smiles_field 是文件中存储smile字符串字段的名称
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH

    def get_data(self):  

        data = self.df
        train_idx = []  # train index
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)] #0.9
        # data2 = data[~data.index.isin(train_idx)] #0.1
        data2 = pd.read_csv("./case.csv")
        print(data2)

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist()) # 如果传入参数是 5 * 2 的矩阵，会生成5个元素，每个一行；一个smile一行，为一个元素
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(256, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(512, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2




    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=self.addH)
        atoms_list = ['<global>'] + atoms_list
        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)  # ???????????????????????????????

        choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1  # permutation(n)，对0到n-1随机排序，返回list  ; +1 不能选到global
        y = np.array(nums_list).astype('int64')  # 原子串本身，相当于label
        weight = np.zeros(len(nums_list))
        for i in choices:
            rand = np.random.rand()  #[0,1)
            weight[i] = 1
            if rand < 0.8:
                nums_list[i] = str2num['<mask>']
            elif rand < 0.9:
                nums_list[i] = int(np.random.rand() * 14 + 1)

        x = np.array(nums_list).astype('int64')  # 遮盖后的原子串，相当于sample
        weight = weight.astype('float32')   # 选到的为1，其余为0
        return x, adjoin_matrix, y, weight



    # map 后放的 预处理函数
    def tf_numerical_smiles(self, data):  
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])  # tf.py_function(python函数, input 由一个或者是几个Tensor组成的list, output一个list或者是tuple)

        x.set_shape([None])  
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])  # 设置为None来保持灵活性并允许任何数量的样本

        return x, adjoin_matrix, y, weight





# ==========================================================================================
'''
test
'''

# train_dataset, test_dataset = Graph_Bert_Dataset(path='D:/TASK2/Data/Drugbank/SMILE/filtered_approved.csv',smiles_field='SMILES',addH=True).get_data()

# for (batch, (x, adjoin_matrix ,y , char_weight)) in enumerate(train_dataset):
#     print(x)
#     print(adjoin_matrix)
#     print(y)
#     print(char_weight)



# def numerical_smiles(smiles):
#         atoms_list, adjoin_matrix = smiles2adjoin(smiles,explicit_hydrogens=True)
#         atoms_list = ['<global>'] + atoms_list
#         nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#         temp = np.ones((len(nums_list),len(nums_list)))
#         temp[1:,1:] = adjoin_matrix
#         adjoin_matrix = (1 - temp) * (-1e9)  # ???????????????????????????????

#         choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1  # permutation(n)，对0到n-1随机排序，返回list  ; +1 不能选到global
#         y = np.array(nums_list).astype('int64')  # 原子串本身，相当于label
#         weight = np.zeros(len(nums_list))
#         for i in choices:
#             rand = np.random.rand()  #[0,1)
#             weight[i] = 1
#             if rand < 0.8:
#                 nums_list[i] = str2num['<mask>']
#             elif rand < 0.9:
#                 nums_list[i] = int(np.random.rand() * 14 + 1)

#         x = np.array(nums_list).astype('int64')  # 遮盖后的原子串，相当于sample
#         weight = weight.astype('float32')   # 选到的为1，其余为0
#         return x, adjoin_matrix, y, weight

# smiless = "[Fe].CC(=O)N(O)CCC[C@@H]1NC(=O)CNC(=O)[C@@H](CC2=CC=CC=C2)NC(=O)CNC(=O)[C@H](CCCN(O)C(C)=O)NC(=O)[C@H](CCCN(O)C(C)=O)NC1=O"

# # x, adjoin_matrix, y, weight = Graph_Bert_Dataset(path='D:/TASK2/Data/Drugbank/SMILE/filtered_approved.csv',smiles_field='SMILES',addH=True).numerical_smiles(smiles)

# x, adjoin_matrix, y, weight = numerical_smiles(smiless)

# print(x)
# print(adjoin_matrix)
# print(y)
# print(weight)

# ==========================================================================================







