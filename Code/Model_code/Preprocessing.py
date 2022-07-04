import pandas as pd
import numpy as np

# =================================================================================

# 将单个药物的输入处理成两个药物的输入：d1?d2

# =================================================================================

# one_drug = pd.read_csv("D:/TASK2/Data/Drugbank/DDI/BIOSNAP/sup_train.csv")



#============================================ 单药物组预训练集 ====================================================


# Drug_ID_1 = []
# Drug_ID_2 = []
# SMILES = []

# # 0 和 3 代表 id 和 smile 字符串的字段
# for i in range(one_drug.shape[0]):
#     print(i)
#     for j in range(one_drug.shape[0]-(i+1)):
#         Drug_ID_1.append(one_drug.iloc[i][0])
#         Drug_ID_2.append(one_drug.iloc[j+i+1][0])
#         SMILES.append(one_drug.iloc[i][3]+"?"+one_drug.iloc[j+i+1][3])


#============================================ Label药物组精调集 ====================================================


# SMILES = []
# for i in range(one_drug.shape[0]):
#     SMILES.append(one_drug.iloc[i][1]+"?"+one_drug.iloc[i][3])


# one_drug["Smiles"] = SMILES

# two_drugs = one_drug

# print(two_drugs)




#============================================ 大文件的 ==============================================================
#     if i==1000 or i==3000 or i==5000 or i==7000 or i==9000:
#         f=open("SMILES.txt","w")
 
#         for line in SMILES:
#             f.write(line+'\n')
#         f.close()


#         f=open("Drug_ID_1.txt","w")
        
#         for line in Drug_ID_1:
#             f.write(line+'\n')
#         f.close()


#         f=open("Drug_ID_2.txt","w")
        
#         for line in Drug_ID_2:
#             f.write(line+'\n')
#         f.close()

# print(Drug_ID_1)
# print(Drug_ID_2)



# f=open("SMILES.txt","w")
 
# for line in SMILES:
#     f.write(line+'\n')
# f.close()


# f=open("Drug_ID_1.txt","w")
 
# for line in Drug_ID_1:
#     f.write(line+'\n')
# f.close()


# f=open("Drug_ID_2.txt","w")
 
# for line in Drug_ID_2:
#     f.write(line+'\n')
# f.close()

#=====================================小文件的===============================================================


# two_drugs = pd.DataFrame({"Drug_ID_1":Drug_ID_1,"Drug_ID_2":Drug_ID_2,"SMILES":SMILES})
# print(two_drugs)


#==============================================================================================================



# two_drugs.to_csv("D:/TASK2/Data/Drugbank/DDI/BIOSNAP/processed_sup_train.csv",index=False)









#================================================ 去除数据集中不能 smiles2adjoin 的 ===============================================================


from utils import smiles2adjoin

before_check = pd.read_csv("clf/DEEPDDI.csv")
# before_check = pd.read_csv("D:/TASK2/Data/DDI/DeepDDI/DEEPDDI.csv")

drop_list = []

for i in range(before_check.shape[0]):
    smiles = before_check.iloc[i][4]
    smiles_1 = smiles.split("?")[0]
    smiles_2 = smiles.split("?")[1]
    atoms_list_1, adjoin_matrix_1 = smiles2adjoin(smiles_1,explicit_hydrogens=True)
    atoms_list_2, adjoin_matrix_2 = smiles2adjoin(smiles_2,explicit_hydrogens=True)
    if atoms_list_1 == "none" or atoms_list_2 == "none":
        drop_list.append(i)
        print(atoms_list_1)
        print(atoms_list_2)

print(drop_list)

after_check = before_check.drop(drop_list)
after_check.to_csv("clf/DEEPDDI_new.csv",index=False)

