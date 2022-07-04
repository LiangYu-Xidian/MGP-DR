from attr import attr
from model import  BertModel, BertModel_test,Drug_BertModel, PredictModel_test
from Dataset_2 import Graph_Bert_Dataset, Drug_Bert_Dataset,Inference_Dataset
import tensorflow as tf
from sklearn.metrics import r2_score,roc_auc_score
import numpy as np

# =================================================================================

# 加载预训练后的模型

# =================================================================================

# small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}
# medium = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'medium_weights','addH':True}
# large = {'name':'Large','num_layers': 12, 'num_heads': 12, 'd_model': 512,'path':'large_weights','addH':True}

# arch = small  ## small 3 4 128   medium: 6 6  256     large:  12 8 516

# num_layers = arch['num_layers']
# num_heads = arch['num_heads']
# d_model = arch['d_model']
# addH = arch['addH']
# dff = d_model * 2
# vocab_size = 17
# dropout_rate = 0.1


# # create dummy/whatever input
# train_dataset, test_dataset = Graph_Bert_Dataset(path='./filtered_approved_2drugs.csv',smiles_field='SMILES',addH=addH).get_data() #p1
# # train_dataset, test_dataset = Graph_Bert_Dataset(path='./S_AB.csv',smiles_field='SMILES',addH=addH).get_data() #p1
# # train_dataset, test_dataset = Drug_Bert_Dataset(path='./S_AB.csv',smiles_field='SMILES',label_field='S_AB',addH=addH).get_data()  #p1+p2


# x, adjoin_matrix , y , char_weight = next(iter(train_dataset.take(1))) #p1
# # x, adjoin_matrix , y , char_weight, s_ab = next(iter(train_dataset.take(1))) #p1+p2

# seq = tf.cast(tf.math.equal(x, 0), tf.float32)
# mask = seq[:, tf.newaxis, tf.newaxis, :]

# # call the model
# temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size) #p1
# # temp = Drug_BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)  #p1+p2

# pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

# temp.load_weights("./Result/Pre/bert_weightsSmall_9.h5")

# temp.summary()

# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# # test_mse = tf.keras.metrics.MeanSquaredError(name='test_mse') # p1+p2


# # p1
# def test_step(x, adjoin_matrix, y, char_weight):  
#     seq = tf.cast(tf.math.equal(x, 0), tf.float32)
#     mask = seq[:, tf.newaxis, tf.newaxis, :]
#     predictions = temp(x,adjoin_matrix=adjoin_matrix,mask=mask,training=False)
#     test_accuracy.update_state(y,predictions,sample_weight=char_weight)

# # p1+p2
# # def test_step(x, adjoin_matrix, y, char_weight,s_ab):
# #     seq = tf.cast(tf.math.equal(x, 0), tf.float32)
# #     mask = seq[:, tf.newaxis, tf.newaxis, :]
# #     predictions_1,predictions_2 = temp(x,adjoin_matrix=adjoin_matrix,mask=mask,training=False)
# #     test_accuracy.update_state(y,predictions_1,sample_weight=char_weight)
# #     test_mse.update_state(s_ab,predictions_2)

            



# for x, adjoin_matrix ,y , char_weight in test_dataset:  #p1
#     test_step(x, adjoin_matrix, y , char_weight)


# # for x, adjoin_matrix ,y , char_weight, s_ab in test_dataset: #p1+p2
# #     test_step(x, adjoin_matrix, y , char_weight, s_ab)


# print('Test Accuracy: {:.4f}'.format(test_accuracy.result()))
# # print('Test MSE: {:.4f}'.format(test_mse.result()))  # p1+p2
# # test_accuracy.reset_states()







# # =================================================================================

# # 加载预训练后的模型进行attention计算

# # =================================================================================

# small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}

# arch = small  ## small 3 4 128   medium: 6 6  256     large:  12 8 516

# num_layers = arch['num_layers']
# num_heads = arch['num_heads']
# d_model = arch['d_model']
# addH = arch['addH']
# dff = d_model * 2
# vocab_size = 17
# dropout_rate = 0.1

# inference_dataset = Inference_Dataset(addH=addH).get_data()

# x, adjoin_matrix, smiles ,atom_list = next(iter(inference_dataset.take(1)))
# seq = tf.cast(tf.math.equal(x, 0), tf.float32)

# mask = seq[:, tf.newaxis, tf.newaxis, :]
# model = BertModel_test(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
# pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
# model.load_weights('./Result/Pre/bert_weightsSmall_9.h5')

# x, adjoin_matrix, smiles ,atom_list = next(iter(inference_dataset.take(1)))
# seq = tf.cast(tf.math.equal(x, 0), tf.float32)
# mask = seq[:, tf.newaxis, tf.newaxis, :]
# x,atts,xs= model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)

# # print(atts)


# from rdkit import Chem
# from IPython.display import SVG,display
# from rdkit.Chem.Draw import rdMolDraw2D
# from rdkit.Chem.Draw.MolDrawing import DrawingOptions

# def plot_weights(smiles,attention_plot,max=5):
#     mol = Chem.MolFromSmiles(smiles)
#     mol = Chem.RemoveHs(mol)
#     num_atoms = mol.GetNumAtoms()
#     atoms = []

#     for i in range(num_atoms):
#         atom = mol.GetAtomWithIdx(i)
#         atoms.append(atom.GetSymbol()+str(i))


#     att = tf.reduce_mean(tf.reduce_mean(attention_plot[:,:,0,:],axis=0),axis=0)[1:].numpy()
#     # att = tf.reduce_mean(tf.reduce_mean(attention_plot[3:,:,0,:],axis=0),axis=0)[1:].numpy()  #num_layers * num_heads * num_atoms * num_atoms
#     print(attention_plot[:,:,0,0].numpy())
#     indices = (-att).argsort()
#     highlight = indices.tolist()
#     print([[atoms[indices[i]],('%.2f'%att[indices[i]])] for i in range(len(indices))])


#     drawer = rdMolDraw2D.MolDraw2DSVG(800,600)
#     opts = drawer.drawOptions()
#     drawer.drawOptions().updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})

    
#     colors = {}
#     for i,h in enumerate(highlight):
#         colors[h] = (1,
#                      1-1*(att[h]-att[highlight[-1]])/(att[highlight[0]]-att[highlight[-1]]),
#                      1-1*(att[h]-att[highlight[-1]])/(att[highlight[0]]-att[highlight[-1]]))
#     drawer.DrawMolecule(mol,highlightAtoms = highlight,highlightAtomColors=colors,highlightBonds=[])
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText().replace('svg:','')
#     with open('./case_study/Amiloride.svg', 'w') as f:
#         f.write(svg)
#     display(SVG(svg))




# i = 1
# smiles_plot = smiles[i].numpy().tolist()[0].decode()
# smiles_1 = smiles_plot.split("?")[0]
# smiles_2 = smiles_plot.split("?")[1]

# mol_1 = Chem.MolFromSmiles(smiles_1)
# mol_2 = Chem.MolFromSmiles(smiles_2)
# num_atoms_1 = mol_1.GetNumAtoms()
# num_atoms_2 = mol_2.GetNumAtoms()
# num_atoms = num_atoms_1 + num_atoms_2
# print(num_atoms)
# print(num_atoms_1)
# print(num_atoms_1)
# attentions_plot_1 = tf.concat([att[i:(i+1),:,:num_atoms_1+1,:num_atoms_1+1] for att in atts],axis=0)
# attentions_plot_2 = tf.concat([att[i:(i+1),:,num_atoms_1+1:num_atoms+1,num_atoms_1+1:num_atoms+1] for att in atts],axis=0)

# print("==========================================")
# print(attentions_plot_1)
# print("==========================================")
# print(smiles_plot)

# plot_weights(smiles_2,attentions_plot_2)






# # =================================================================================

# # 加载分类后的模型

# # =================================================================================

# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.layers as layers
# import pandas as pd
# import numpy as np

# from Dataset_2 import Graph_Classification_Dataset
# from sklearn.metrics import r2_score,roc_auc_score,precision_recall_curve,auc,f1_score,precision_score,recall_score

# from model import  PredictModel,BertModel,Drug_BertModel


# keras.backend.clear_session()

# def main(seed):
    
#     task = 'DCDB'
#     print(task)

#     small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}
#     medium = {'name':'Medium','num_layers': 3, 'num_heads': 2, 'd_model': 256,'path':'medium_weights','addH':True}

#     arch = small  ## small 3 2 128   medium: 6 4  256     large:  12 8 512

#     pretraining = True
#     pretraining_str = 'pretraining' if pretraining else ''

#     trained_epoch = 10

#     num_layers = arch['num_layers']
#     num_heads = arch['num_heads']
#     d_model = arch['d_model']
#     addH = arch['addH']
#     dff = d_model * 2
#     vocab_size = 17
#     dropout_rate = 0.1

#     seed = seed
#     np.random.seed(seed=seed)
#     tf.random.set_seed(seed=seed)
#     train_dataset, test_dataset , val_dataset = Graph_Classification_Dataset('clf/{}.csv'.format(task), smiles_field='Smiles',
#                                                                label_field='Label',addH=addH).get_data()

#     x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
#     seq = tf.cast(tf.math.equal(x, 0), tf.float32)
#     mask = seq[:, tf.newaxis, tf.newaxis, :]
#     model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
#                          dense_dropout=0.5)

#     if pretraining:
#         temp = Drug_BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
#         # temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
#         pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

#         temp.load_weights("./Result/Pre/bert_weightsLarge_128_s_ab_s1_10.h5")
#         # temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))

#         temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
#         del temp

#         pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
#         model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
#         print('load_wieghts')


#     y_true = []
#     y_preds = []
#     model.load_weights('Result/DCDB/DCDB_100.h5')
#     print('load_clf_wieghts')
#     print(model.summary())
#     for x, adjoin_matrix, y in test_dataset:
#         seq = tf.cast(tf.math.equal(x, 0), tf.float32)
#         mask = seq[:, tf.newaxis, tf.newaxis, :]
#         preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
#         y_true.append(y.numpy())
#         y_preds.append(preds.numpy())
#     y_true = np.concatenate(y_true, axis=0).reshape(-1)
#     y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
#     y_preds = tf.sigmoid(y_preds).numpy()

#     print(y_true,y_preds)
#     result = pd.DataFrame({"y_true":y_true,"y_preds":y_preds})
#     result.to_csv("ValidResult/{}_{}.csv".format(task,seed),index=False)

#     test_auc = roc_auc_score(y_true, y_preds)
#     test_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
#     prec, reca, _ = precision_recall_curve(y_true, y_preds)
#     test_aupr = auc(reca, prec)
#     print(test_aupr)
#     # test_aupr = keras.metrics.(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
#     test_f1 = f1_score(y_true.reshape(-1), y_preds.reshape(-1).round())
#     print(test_f1)
#     test_precision = precision_score(y_true,np.around(y_preds,0).astype(int))
#     test_recall = recall_score(y_true,np.around(y_preds,0).astype(int))
#     print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_accuracy), 'test aupr:{:.4f}'.format(test_aupr), 'test f1:{:.4f}'.format(test_f1), 'test precision:{:.4f}'.format(test_precision), 'test recall:{:.4f}'.format(test_recall))

#     return test_auc

# if __name__ == '__main__':

#     auc_list = []
#     for seed in [100]: # 6,8,10   66,100,120    121(256后的)
#         print(seed)
#         auc = main(seed)
#         auc_list.append(auc)
#     print(auc_list)














# =================================================================================

# 加载多分类后的模型

# =================================================================================

# import tensorflow as tf
# import tensorflow.keras as keras
# import tensorflow.keras.layers as layers
# import pandas as pd
# import numpy as np

# from Dataset_2 import Graph_multi_Classification_Dataset
# from sklearn.metrics import r2_score,roc_auc_score,auc,accuracy_score, precision_recall_curve, precision_score,recall_score,f1_score
# from sklearn.preprocessing import label_binarize
# from sklearn import preprocessing

# from model import  Multi_PredictModel,BertModel


# keras.backend.clear_session()

# def main(seed):
    
#     task = 'DEEPDDI'
#     print(task)

#     small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}
#     medium = {'name':'Medium','num_layers': 3, 'num_heads': 2, 'd_model': 256,'path':'medium_weights','addH':True}

#     arch = small  ## small 3 2 128   medium: 6 4  256     large:  12 8 512

#     pretraining = True
#     pretraining_str = 'pretraining' if pretraining else ''

#     trained_epoch = 9

#     num_layers = arch['num_layers']
#     num_heads = arch['num_heads']
#     d_model = arch['d_model']
#     addH = arch['addH']
#     dff = d_model * 2
#     vocab_size = 17
#     dropout_rate = 0.1

#     seed = seed
#     np.random.seed(seed=seed)
#     tf.random.set_seed(seed=seed)
#     train_dataset, test_dataset , val_dataset = Graph_multi_Classification_Dataset('clf/{}.csv'.format(task), smiles_field='Smiles',
#                                                                label_field='Label',addH=addH).get_data()

#     x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
#     seq = tf.cast(tf.math.equal(x, 0), tf.float32)
#     mask = seq[:, tf.newaxis, tf.newaxis, :]
#     model = Multi_PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
#                          dense_dropout=0)

#     if pretraining:
#         temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
#         pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
#         temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
#         temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
#         del temp

#         pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
#         model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
#         print('load_wieghts')


#     y_true = []
#     y_preds = []

#     flag = 0

#     model.load_weights('Result/DEEPDDI/{}_{}.h5'.format(task, seed))
#     print('load_clf_wieghts')
#     print(model.summary())

#     for x, adjoin_matrix, y in test_dataset:
#         seq = tf.cast(tf.math.equal(x, 0), tf.float32)
#         mask = seq[:, tf.newaxis, tf.newaxis, :]
#         preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
#         y_true.append(y.numpy())
#         y_preds.append(tf.sigmoid(preds).numpy().argmax(axis = 1).tolist())

#         if flag == 0:
#             y_score = tf.sigmoid(preds).numpy()
#             flag = 1
#         else:
#             y_score = np.append(np.array(y_score),tf.sigmoid(preds).numpy(),axis=0)
#     y_score = (y_score.T/y_score.sum(axis=1)).T

#     y_true = np.concatenate(y_true, axis=0).reshape(-1)
#     y_one_hot = label_binarize(y_true, np.arange(86))
#     y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
#     y_score = np.array(y_score)

#     test_acc = accuracy_score(y_true, y_preds)
#     test_auc = roc_auc_score(y_one_hot, y_score,multi_class="ovr")



#     with open("pr_curve.txt", 'w') as pr:
#         for i in range(y_score.shape[0]):
#             scores = ""
#             for j in range(86):
#                 scores = scores + str(format(y_score[i][j], '.4f')) + " "
#             scores = scores.rstrip() + "\n"
#             pr.write(str(y_true[i]) + " " + str(y_preds[i]) + " " + str(scores))


#     print(y_true)
#     print(y_preds)
#     print(y_score)
#     print(y_true.shape)
#     print(y_preds.shape)
#     print(y_score.shape)

#     # macro_recall = recall_score(y_true,y_preds,average="macro")
#     # macro_precision = precision_score(y_true,y_preds,average="macro")
#     # macro_f1 = f1_score(y_true,y_preds,average="macro")
#     # micro_recall = recall_score(y_true,y_preds,average="micro")
#     # micro_precision = precision_score(y_true,y_preds,average="micro")
#     # micro_f1 = f1_score(y_true,y_preds,average="micro")

#     # prec, reca, _ = precision_recall_curve(y_true, y_preds,pos_label=0)
#     # test_aupr = auc(reca, prec)

#     mean_aupr = 0
#     for i in range(86):
#         prec, reca, _ = precision_recall_curve(y_true, y_preds,pos_label=i)
#         test_aupr = auc(reca, prec)
#         mean_aupr = mean_aupr + test_aupr
    
#     print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_acc), 'test aupr:{:.4f}'.format(test_aupr))

#     return test_auc

# if __name__ == '__main__':

#     auc_list = []
#     for seed in [999]: 
#         print(seed)
#         auc = main(seed)
#         auc_list.append(auc)
#     print(auc_list)















# # =================================================================================

# # 加载分类后的模型 visualization

# # =================================================================================

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')

from Dataset_2 import Graph_Classification_Dataset
from sklearn.metrics import r2_score,roc_auc_score

from model import  PredictModel,BertModel,Drug_BertModel


keras.backend.clear_session()

def main(seed):
    
    task = 'DCDB'  #############################################################
    print(task)

    small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}

    arch = small  ## small 3 2 128   medium: 6 4  256     large:  12 8 512

    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 10

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']
    dff = d_model * 2
    vocab_size = 17
    dropout_rate = 0.1

    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset , val_dataset = Graph_Classification_Dataset('clf/{}.csv'.format(task), smiles_field='Smiles',
                                                               label_field='Label',addH=addH).get_data()

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel_test(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.5)

    if pretraining:
        temp = BertModel_test(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

        temp.load_weights("./Result/Pre/bert_weightsSmall_9.h5")

        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')


    
    model.load_weights('Result/DCDB/DCDB_18.h5') #####################################################
    print('load_clf_wieghts')
    print(model.summary())
  
    nums = 0
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds,SUP_embedding = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)

        y = y.numpy()
        SUP_embedding = SUP_embedding.numpy()

        
        # SUP_embedding_all = SUP_embedding_all + SUP_embedding
        if nums == 0:
            SUP_embedding_all = SUP_embedding
            y_all = y
        else:
            SUP_embedding_all = np.vstack((SUP_embedding_all,SUP_embedding))
            y_all  = np.vstack((y_all,y))
        
        nums = nums + 1
        
    print(y_all.shape)
    print(SUP_embedding_all.shape)


    # tsne 画图
    embeddings_list = SUP_embedding_all
    tsne = TSNE()
    print('start to process')
    Y = tsne.fit_transform(np.vstack(embeddings_list))
    index_1 = []
    index_0 = []
    for i in range(Y.shape[0]):
        if y_all[i][0] == 1:
            index_1.append(i)
        if y_all[i][0] == 0:
            index_0.append(i)
    Y_1 = np.delete(Y,index_0,axis=0)
    Y_0 = np.delete(Y,index_1,axis=0)
    print('Done')
    print(len(Y_1))
    print(len(Y_0))
    print(Y_0)

    plt.scatter(Y_1[:, 0], Y_1[:, 1], label = "Positive", s = 13)
    plt.scatter(Y_0[:, 0], Y_0[:, 1], label = "Negative", s = 13)
    plt.legend()
    plt.title("Visualization of Representation Learned from DCDB")

    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)

    plt.xlabel('tSNE-1')
    plt.ylabel('tSNE-2')


    plt.savefig("./Result/Visualization/DCDB_18.png") ##############################



        
   


if __name__ == '__main__':

    for seed in [18]: # 6,8,10   66,100,120    121(256后的) #################################
        print(seed)
        auc = main(seed)