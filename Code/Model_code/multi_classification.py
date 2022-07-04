import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

from Dataset_2 import Graph_multi_Classification_Dataset
from sklearn.metrics import r2_score,roc_auc_score,precision_score,recall_score,f1_score,accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing

from model import  Multi_PredictModel,BertModel


keras.backend.clear_session()


def main(seed):

    task = 'DEEPDDI'
    print(task)

    small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}

    arch = small  ## small 3 2 128   medium: 6 4  256     large:  12 8 512

    pretraining = False
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 9

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
    train_dataset, test_dataset , val_dataset = Graph_multi_Classification_Dataset('clf/{}.csv'.format(task), smiles_field='Smiles',
                                                               label_field='Label',addH=addH).get_data()
    print("--------------------------------------------------------- Data over -----------------------------------------------------------")

    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = Multi_PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')


    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) 

    acc= -10
    auc= -10
    stopping_monitor = 0
    for epoch in range(250):
        accuracy_object = tf.keras.metrics.SparseCategoricalAccuracy()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        for x,adjoin_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                loss = loss_object(y,preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                y_preds = tf.sigmoid(preds).numpy()
                accuracy_object.update_state(y,y_preds)
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()),'accuracy: {:.4f}'.format(accuracy_object.result().numpy().item()))

        y_true = []
        y_preds = []
        
        flag = 0

        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(tf.sigmoid(preds).numpy().argmax(axis = 1).tolist())

            if flag == 0:
                y_score = tf.sigmoid(preds).numpy()
                flag = 1
            else:
                y_score = np.append(np.array(y_score),tf.sigmoid(preds).numpy(),axis=0)
        
        y_score = (y_score.T/y_score.sum(axis=1)).T

        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_one_hot = label_binarize(y_true, np.arange(86))
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)

        y_score = np.array(y_score)

        acc_new = accuracy_score(y_true,y_preds)
        auc_new = roc_auc_score(y_one_hot,y_score,multi_class="ovr")
        
        macro_recall = recall_score(y_true,y_preds,average="macro")
        macro_precision = precision_score(y_true,y_preds,average="macro")
        macro_f1 = f1_score(y_true,y_preds,average="macro")
        micro_recall = recall_score(y_true,y_preds,average="micro")
        micro_precision = precision_score(y_true,y_preds,average="micro")
        micro_f1 = f1_score(y_true,y_preds,average="micro")
        print('val acc:{:.4f}'.format(acc_new))
        print('val auc:{:.4f}'.format(auc_new))
        print('val macro_recall:{:.4f}'.format(macro_recall),'val macro_precision:{:.4f}'.format(macro_precision),'val macro_f1:{:.4f}'.format(macro_f1))
        print('val micro_recall:{:.4f}'.format(micro_recall),'val micro_precision:{:.4f}'.format(micro_precision),'val micro_f1:{:.4f}'.format(micro_f1))

        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch,pretraining_str),
                    [y_true, y_preds])
            model.save_weights('classification_weights/{}_{}.h5'.format(task,seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor>0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor>20:
            break

    y_true = []
    y_preds = []

    flag = 0

    model.load_weights('classification_weights/{}_{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(tf.sigmoid(preds).numpy().argmax(axis = 1).tolist())

        if flag == 0:
            y_score = tf.sigmoid(preds).numpy()
            flag = 1
        else:
            y_score = np.append(np.array(y_score),tf.sigmoid(preds).numpy(),axis=0)
    y_score = (y_score.T/y_score.sum(axis=1)).T

    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_one_hot = label_binarize(y_true, np.arange(86))
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    y_score = np.array(y_score)

    test_acc = accuracy_score(y_true, y_preds)
    test_auc = roc_auc_score(y_one_hot, y_score,multi_class="ovr")
    
    macro_recall = recall_score(y_true,y_preds,average="macro")
    macro_precision = precision_score(y_true,y_preds,average="macro")
    macro_f1 = f1_score(y_true,y_preds,average="macro")
    micro_recall = recall_score(y_true,y_preds,average="micro")
    micro_precision = precision_score(y_true,y_preds,average="micro")
    micro_f1 = f1_score(y_true,y_preds,average="micro")
    print('test acc:{:.4f}'.format(test_acc))
    print('test auc:{:.4f}'.format(test_auc))
    print('test macro_recall:{:.4f}'.format(macro_recall),'test macro_precision:{:.4f}'.format(macro_precision),'test macro_f1:{:.4f}'.format(macro_f1))
    print('test micro_recall:{:.4f}'.format(micro_recall),'test micro_precision:{:.4f}'.format(micro_precision),'test micro_f1:{:.4f}'.format(micro_f1))

    return test_auc

if __name__ == '__main__':

    acc_list = []
    for seed in [999]:
        print(seed)
        acc = main(seed)
        acc_list.append(acc)
    print(acc_list)
