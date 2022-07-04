# -*- coding: utf-8 -*-
from numpy.core import numeric
import pandas as pd
import numpy as np
from utils import smiles2adjoin
import tensorflow as tf
import scipy.linalg
from sklearn.utils import shuffle

# =============================================================================================================================================

# Strategy_1：two drug one supernode
#    
# Strategy_2：one drug one supernode, a super-supernode connect two supernode
#    

# =============================================================================================================================================


# DrugBank dict
str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
         'I': 10,'Na': 11,'Fe':12,'Mg':13,'<unk>':14,'<mask>':15,'<global>':16}

num2str =  {i:j for j,i in str2num.items()}




# ================================================== Graph_Bert_Dataset ===========================================================================================


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
        data2 = data[~data.index.isin(train_idx)] #0.1
        # data2 = pd.read_csv("./case.csv")
        # print("--------------------------------Dataset_2-------------------------------------------------------")
        # print(data2)

        self.dataset1 = tf.data.Dataset.from_tensor_slices(data1[self.smiles_field].tolist()) # 如果传入参数是 5 * 2 的矩阵，会生成5个元素，每个一行；这里一个smile一行，为一个元素
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]))).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices(data2[self.smiles_field].tolist())
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2




    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        smiles_1 = smiles.split("?")[0]
        smiles_2 = smiles.split("?")[1]
        atoms_list_1, adjoin_matrix_1 = smiles2adjoin(smiles_1,explicit_hydrogens=self.addH)
        atoms_list_2, adjoin_matrix_2 = smiles2adjoin(smiles_2,explicit_hydrogens=self.addH)


        # Strategy_1
        atoms_list = atoms_list_1 + atoms_list_2
        atoms_list = ['<global>'] + atoms_list
        adjoin_matrix = scipy.linalg.block_diag(adjoin_matrix_1, adjoin_matrix_2)

        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)  

        # Strategy_2
        # atoms_list = ['<global>'] + list(atoms_list_1) + ['<global>'] + list(atoms_list_2)
        # atoms_list = ['<global>'] + atoms_list

        # temp_1 = np.ones((len(atoms_list_1)+1,len(atoms_list_1)+1))
        # temp_1[1:,1:] = adjoin_matrix_1
        # temp_2 = np.ones((len(atoms_list_2)+1,len(atoms_list_2)+1))
        # temp_2[1:,1:] = adjoin_matrix_2
        # adjoin_matrix = scipy.linalg.block_diag(temp_1, temp_2)

        # nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        # temp = np.ones((len(nums_list),len(nums_list)))
        # temp[1:,1:] = adjoin_matrix
        # adjoin_matrix = (1 - temp) * (-1e9)  
        

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



    # map 后放的 预处理函数, getdata()调用它
    def tf_numerical_smiles(self, data):  
        x, adjoin_matrix, y, weight = tf.py_function(self.numerical_smiles, [data],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32])  # tf.py_function(python函数, input 由一个或者是几个Tensor组成的list, output一个list或者是tuple)

        x.set_shape([None])  
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])  # 设置为None来保持灵活性并允许任何数量的样本

        return x, adjoin_matrix, y, weight





# ========================================================== test ========================================================================================


# train_dataset, test_dataset = Graph_Bert_Dataset(path='D:/TASK2/Data/Drugbank/SMILE/filtered_approved.csv',smiles_field='SMILES',addH=True).get_data()

# for (batch, (x, adjoin_matrix ,y , char_weight)) in enumerate(train_dataset):
#     print(x)
#     print(adjoin_matrix)
#     print(y)
#     print(char_weight)


# smiles = "[Fe].CC(=O)N(O)CCC[C@@H]1NC(=O)CNC(=O)[C@@H](CC2=CC=CC=C2)NC(=O)CNC(=O)[C@H](CCCN(O)C(C)=O)NC(=O)[C@H](CCCN(O)C(C)=O)NC1=O"

# x, adjoin_matrix, y, weight = Graph_Bert_Dataset(path='D:/TASK2/Data/Drugbank/SMILE/filtered_approved.csv',smiles_field='SMILES',addH=True).numerical_smiles(smiles)

# print(x)
# print(adjoin_matrix)
# print(y)
# print(weight)



#=================================================== two drug test =====================================================================================

# def numerical_smiles(smiles):
#     # smiles = smiles.numpy().decode()
#     smiles_1 = smiles.split("?")[0]
#     smiles_2 = smiles.split("?")[1]
#     atoms_list_1, adjoin_matrix_1 = smiles2adjoin(smiles_1,explicit_hydrogens=True)
#     atoms_list_2, adjoin_matrix_2 = smiles2adjoin(smiles_2,explicit_hydrogens=True)


#     # Strategy_1
#     atoms_list = atoms_list_1 + atoms_list_2
#     atoms_list = ['<global>'] + atoms_list
#     print(atoms_list_1)
#     print(atoms_list_2)
#     print(atoms_list)
#     adjoin_matrix = scipy.linalg.block_diag(adjoin_matrix_1, adjoin_matrix_2)
#     print(adjoin_matrix_1)
#     print(adjoin_matrix_2)
#     print(adjoin_matrix)

#     nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
#     temp = np.ones((len(nums_list),len(nums_list)))
#     temp[1:,1:] = adjoin_matrix
#     adjoin_matrix = (1 - temp) * (-1e9)  
#     print(adjoin_matrix)
    

#     choices = np.random.permutation(len(nums_list)-1)[:max(int(len(nums_list)*0.15),1)] + 1  # permutation(n)，对0到n-1随机排序，返回list  ; +1 不能选到global
#     y = np.array(nums_list).astype('int64')  # 原子串本身，相当于label
#     weight = np.zeros(len(nums_list))
#     for i in choices:
#         rand = np.random.rand()  #[0,1)
#         weight[i] = 1
#         if rand < 0.8:
#             nums_list[i] = str2num['<mask>']
#         elif rand < 0.9:
#             nums_list[i] = int(np.random.rand() * 14 + 1)

#     x = np.array(nums_list).astype('int64')  # 遮盖后的原子串，相当于sample
#     weight = weight.astype('float32')   # 选到的为1，其余为0
#     return x, adjoin_matrix, y, weight


# smiles = "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O?CC(C)NCC(O)COC1=CC=C(CCOCC2CC2)C=C1" # drug1?drug2
# x, adjoin_matrix, y, weight = numerical_smiles(smiles)
# print(x)
# print(y)
# print(adjoin_matrix)
# print(weight)

# ==========================================================================================================================================================
# ==========================================================================================================================================================







# ============================================================ Graph_Classification_Dataset ==============================================================================================



class Graph_Classification_Dataset(object):
    def __init__(self,path,smiles_field='Smiles',label_field='Label',addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        # self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH

    def get_data(self):
        data = self.df
        data = data.dropna()
        data[self.label_field] = data[self.label_field].map(int)


        #=================================== 没有分数据集的处理方式 ============================
        pdata = data[data[self.label_field] == 1]
        ndata = data[data[self.label_field] == 0]
        

        ptrain_idx = []
        ptrain_idx = pdata.sample(frac=0.8).index
        p_train_data = pdata[pdata.index.isin(ptrain_idx)]

        ntrain_idx = []
        ntrain_idx = ndata.sample(frac=0.8).index
        n_train_data = ndata[ndata.index.isin(ntrain_idx)]

        train_data = p_train_data.append(n_train_data)
        train_data = shuffle(train_data)
        print(train_data)


        pdata = pdata[~pdata.index.isin(ptrain_idx)]
        ndata = ndata[~ndata.index.isin(ntrain_idx)]


        ptest_idx = []
        ptest_idx = pdata.sample(frac=0.5).index
        p_test_data = pdata[pdata.index.isin(ptest_idx)]

        ntest_idx = []
        ntest_idx = ndata.sample(frac=0.5).index
        n_test_data = ndata[ndata.index.isin(ntest_idx)]

        test_data = p_test_data.append(n_test_data)
        test_data = shuffle(test_data)
        print(test_data)
        # test_data.to_csv("ValidResult/DCDB.csv")
        

        p_val_data = pdata[~pdata.index.isin(ptest_idx)]
        n_val_data = ndata[~ndata.index.isin(ntest_idx)]
        val_data = p_val_data.append(n_val_data)
        val_data = shuffle(val_data)
        print(val_data)

        #=================================== CASTER 的数据 ============================

        # test_idx = [i for i in range(16608)]
        # test_data = data[data.index.isin(test_idx)]
        # test_data = shuffle(test_data)
        # print(test_data)
        # test_data.to_csv("ValidResult/BIOSNAP.csv")
        

        # data = data[~data.index.isin(test_idx)]

        # pdata = data[data[self.label_field] == 1]
        # ndata = data[data[self.label_field] == 0]
        

        # ptrain_idx = []
        # ptrain_idx = pdata.sample(frac=0.8).index
        # p_train_data = pdata[pdata.index.isin(ptrain_idx)]

        # ntrain_idx = []
        # ntrain_idx = ndata.sample(frac=0.8).index
        # n_train_data = ndata[ndata.index.isin(ntrain_idx)]

        # train_data = p_train_data.append(n_train_data)
        # train_data = shuffle(train_data)
        # print(train_data)


        # p_val_data = pdata[~pdata.index.isin(ptrain_idx)]
        # n_val_data = ndata[~ndata.index.isin(ntrain_idx)]
        # val_data = p_val_data.append(n_val_data)
        # val_data = shuffle(val_data)
        # print(val_data)



        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(32, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).prefetch(100)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(128, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).padded_batch(128, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).cache().prefetch(100)

        return self.dataset1, self.dataset2, self.dataset3




    def numerical_smiles(self, smiles,label):
        smiles = smiles.numpy().decode()
        smiles_1 = smiles.split("?")[0]
        smiles_2 = smiles.split("?")[1]
        atoms_list_1, adjoin_matrix_1 = smiles2adjoin(smiles_1,explicit_hydrogens=self.addH)
        atoms_list_2, adjoin_matrix_2 = smiles2adjoin(smiles_2,explicit_hydrogens=self.addH)

        # Strategy_1
        atoms_list = atoms_list_1 + atoms_list_2
        atoms_list = ['<global>'] + atoms_list
        adjoin_matrix = scipy.linalg.block_diag(adjoin_matrix_1, adjoin_matrix_2)

        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)

        # Strategy_2
        # atoms_list = ['<global>'] + list(atoms_list_1) + ['<global>'] + list(atoms_list_2)
        # atoms_list = ['<global>'] + atoms_list

        # temp_1 = np.ones((len(atoms_list_1)+1,len(atoms_list_1)+1))
        # temp_1[1:,1:] = adjoin_matrix_1
        # temp_2 = np.ones((len(atoms_list_2)+1,len(atoms_list_2)+1))
        # temp_2[1:,1:] = adjoin_matrix_2
        # adjoin_matrix = scipy.linalg.block_diag(temp_1, temp_2)

        # nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        # temp = np.ones((len(nums_list),len(nums_list)))
        # temp[1:,1:] = adjoin_matrix
        # adjoin_matrix = (1 - temp) * (-1e9) 


        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('int64')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y





# ============================================================ Graph_multi_Classification_Dataset ==============================================================================================



class Graph_multi_Classification_Dataset(object):
    def __init__(self,path,smiles_field='Smiles',label_field='Label',addH=True):
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        # self.df = self.df[self.df[smiles_field].str.len() <= max_len]
        self.addH = addH

    def get_data(self):
        data = self.df
        data = data.dropna()
        data[self.label_field] = data[self.label_field].map(int)  
        data[self.label_field] = data[self.label_field] - 1      # DEEPDDI
        
        #=================================== DEEPDDI 的数据 ============================

        train_idx = []
        test_idx = []
        val_idx = []
        for i in range(data.shape[0]):
            if data.iloc[i][3] == "training":
                train_idx.append(i)
            if data.iloc[i][3] == "testing":
                test_idx.append(i)
            if data.iloc[i][3] == "validation":
                val_idx.append(i)
            

        train_data = data[data.index.isin(train_idx)]
        train_data = shuffle(train_data)
        print(train_data)
        
        test_data = data[data.index.isin(test_idx)]
        test_data = shuffle(test_data)
        print(test_data)
        
        val_data = data[data.index.isin(val_idx)]
        val_data = shuffle(val_data)
        print(val_data)



        self.dataset1 = tf.data.Dataset.from_tensor_slices(
            (train_data[self.smiles_field], train_data[self.label_field]))
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).cache().padded_batch(128, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).prefetch(100)
        
        print("-------------------------------------------------dataset1--------------------------------------------------------")

        self.dataset2 = tf.data.Dataset.from_tensor_slices((test_data[self.smiles_field], test_data[self.label_field]))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).cache().padded_batch(128, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).prefetch(100)
        
        print("-------------------------------------------------dataset2--------------------------------------------------------")

        self.dataset3 = tf.data.Dataset.from_tensor_slices((val_data[self.smiles_field], val_data[self.label_field]))
        self.dataset3 = self.dataset3.map(self.tf_numerical_smiles).cache().padded_batch(128, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([1]))).prefetch(100)

        print("-------------------------------------------------dataset3--------------------------------------------------------")

        return self.dataset1, self.dataset2, self.dataset3




    def numerical_smiles(self, smiles,label):
        smiles = smiles.numpy().decode()
        smiles_1 = smiles.split("?")[0]
        smiles_2 = smiles.split("?")[1]
        atoms_list_1, adjoin_matrix_1 = smiles2adjoin(smiles_1,explicit_hydrogens=self.addH)
        atoms_list_2, adjoin_matrix_2 = smiles2adjoin(smiles_2,explicit_hydrogens=self.addH)

        # Strategy_1
        atoms_list = list(atoms_list_1) + list(atoms_list_2)
        print("----------------------1,2,1+2------------------------")
        print(smiles)
        atoms_list = ['<global>'] + atoms_list
        adjoin_matrix = scipy.linalg.block_diag(adjoin_matrix_1, adjoin_matrix_2)

        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        print(nums_list)
        print(len(nums_list))
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)
        x = np.array(nums_list).astype('int64')
        y = np.array([label]).astype('int64')
        return x, adjoin_matrix,y

    def tf_numerical_smiles(self, smiles,label):
        x,adjoin_matrix,y = tf.py_function(self.numerical_smiles, [smiles,label], [tf.int64, tf.float32 ,tf.int64])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        return x, adjoin_matrix , y







# ================================================== Drug_Bert_Dataset ===========================================================================================


class Drug_Bert_Dataset(object):
    def __init__(self,path,smiles_field='SMILES',label_field='S_AB',normalize=False,addH=True):  # smiles_field 是文件中存储smile字符串字段的名称
        if path.endswith('.txt') or path.endswith('.tsv'):
            self.df = pd.read_csv(path,sep='\t')
        else:
            self.df = pd.read_csv(path)
        self.smiles_field = smiles_field
        self.label_field = label_field
        self.vocab = str2num
        self.devocab = num2str
        self.addH = addH
        if normalize:
            self.max = self.df[self.label_field].max()
            self.min = self.df[self.label_field].min()
            self.df[self.label_field] = (self.df[self.label_field]-self.min)/(self.max-self.min) #最大最小归一化

    def get_data(self):  

        data = self.df
        train_idx = []  # train index
        idx = data.sample(frac=0.9).index
        train_idx.extend(idx)

        data1 = data[data.index.isin(train_idx)] #0.9
        data2 = data[~data.index.isin(train_idx)] #0.1

        self.dataset1 = tf.data.Dataset.from_tensor_slices((data1[self.smiles_field].tolist(),data1[self.label_field].tolist())) # 如果传入参数是 5 * 2 的矩阵，会生成5个元素，每个一行；这里一个smile一行，为一个元素
        self.dataset1 = self.dataset1.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]),tf.TensorShape([None,None]), tf.TensorShape([None]) ,tf.TensorShape([None]),tf.TensorShape([None]))).prefetch(50)

        self.dataset2 = tf.data.Dataset.from_tensor_slices((data2[self.smiles_field].tolist(),data2[self.label_field].tolist()))
        self.dataset2 = self.dataset2.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None, None]), tf.TensorShape([None]),
            tf.TensorShape([None]),tf.TensorShape([None]))).prefetch(50)
        return self.dataset1, self.dataset2




    def numerical_smiles(self, smiles,label):
        smiles = smiles.numpy().decode()
        smiles_1 = smiles.split("?")[0]
        smiles_2 = smiles.split("?")[1]
        atoms_list_1, adjoin_matrix_1 = smiles2adjoin(smiles_1,explicit_hydrogens=self.addH)
        atoms_list_2, adjoin_matrix_2 = smiles2adjoin(smiles_2,explicit_hydrogens=self.addH)


        # Strategy_1
        # atoms_list = atoms_list_1 + atoms_list_2
        # atoms_list = ['<global>'] + atoms_list
        # adjoin_matrix = scipy.linalg.block_diag(adjoin_matrix_1, adjoin_matrix_2)

        # nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        # temp = np.ones((len(nums_list),len(nums_list)))
        # temp[1:,1:] = adjoin_matrix
        # adjoin_matrix = (1 - temp) * (-1e9)  

        # Strategy_2
        atoms_list = ['<global>'] + list(atoms_list_1) + ['<global>'] + list(atoms_list_2)
        atoms_list = ['<global>'] + atoms_list

        temp_1 = np.ones((len(atoms_list_1)+1,len(atoms_list_1)+1))
        temp_1[1:,1:] = adjoin_matrix_1
        temp_2 = np.ones((len(atoms_list_2)+1,len(atoms_list_2)+1))
        temp_2[1:,1:] = adjoin_matrix_2
        adjoin_matrix = scipy.linalg.block_diag(temp_1, temp_2)

        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:,1:] = adjoin_matrix
        adjoin_matrix = (1 - temp) * (-1e9)  
        

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
        s_ab = np.array([label]).astype('float32')
        return x, adjoin_matrix, y, weight, s_ab



    # map 后放的 预处理函数, getdata()调用它
    def tf_numerical_smiles(self, data,label):  
        x, adjoin_matrix, y, weight, s_ab = tf.py_function(self.numerical_smiles, [data,label],
                                                     [tf.int64, tf.float32, tf.int64, tf.float32, tf.float32])  # tf.py_function(python函数, input 由一个或者是几个Tensor组成的list, output一个list或者是tuple)

        x.set_shape([None])  
        adjoin_matrix.set_shape([None,None])
        y.set_shape([None])
        weight.set_shape([None])  # 设置为None来保持灵活性并允许任何数量的样本
        s_ab.set_shape([None])

        return x, adjoin_matrix, y, weight, s_ab



# ========================================================== test ========================================================================================






# ============================================================ Inference_Dataset ==============================================================================================

class Inference_Dataset(object):
    def __init__(self,max_len=100,addH=True):
        self.vocab = str2num
        self.devocab = num2str
        # self.sml_list = [i for i in sml_list if len(i)<max_len]
        self.addH =  addH

    def get_data(self):

        data = pd.read_csv("./case.csv")
        print("--------------------------------Dataset_2-------------------------------------------------------")
        print(data)

        self.dataset = tf.data.Dataset.from_tensor_slices(data["SMILES"].tolist())
        self.dataset = self.dataset.map(self.tf_numerical_smiles).padded_batch(64, padded_shapes=(
            tf.TensorShape([None]), tf.TensorShape([None,None]),tf.TensorShape([1]),tf.TensorShape([None]))).cache().prefetch(20)

        return self.dataset

    def numerical_smiles(self, smiles):
        smiles = smiles.numpy().decode()
        smiles_1 = smiles.split("?")[0]
        smiles_2 = smiles.split("?")[1]
        atoms_list_1, adjoin_matrix_1 = smiles2adjoin(smiles_1,explicit_hydrogens=self.addH)
        atoms_list_2, adjoin_matrix_2 = smiles2adjoin(smiles_2,explicit_hydrogens=self.addH)

        # Strategy_1
        atoms_list = atoms_list_1 + atoms_list_2
        atoms_list = ['<global>'] + atoms_list
        adjoin_matrix = scipy.linalg.block_diag(adjoin_matrix_1, adjoin_matrix_2)

        nums_list =  [str2num.get(i,str2num['<unk>']) for i in atoms_list]
        temp = np.ones((len(nums_list),len(nums_list)))
        temp[1:, 1:] = adjoin_matrix
        adjoin_matrix = (1-temp)*(-1e9)


        x = np.array(nums_list).astype('int64')

        return x, adjoin_matrix,[smiles], atoms_list

    def tf_numerical_smiles(self, smiles):
        x,adjoin_matrix,smiles,atom_list = tf.py_function(self.numerical_smiles, [smiles], [tf.int64, tf.float32,tf.string, tf.string])
        x.set_shape([None])
        adjoin_matrix.set_shape([None,None])
        smiles.set_shape([1])
        atom_list.set_shape([None])
        return x, adjoin_matrix,smiles,atom_list
