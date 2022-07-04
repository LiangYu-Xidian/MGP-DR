import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

from Dataset_2_clf import Graph_Classification_Dataset
from sklearn.metrics import r2_score,roc_auc_score

from model import  PredictModel,BertModel,Drug_BertModel


keras.backend.clear_session()

def main(seed):

    task = 'GCN_BMP'
    print(task)

    small = {'name':'Small','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'small_weights','addH':True}
    medium = {'name':'Medium','num_layers': 3, 'num_heads': 2, 'd_model': 256,'path':'medium_weights','addH':True}
    large = {'name':'Large','num_layers': 3, 'num_heads': 2, 'd_model': 128,'path':'large_weights','addH':True}

    arch = large  ## small 3 2 128   medium: 6 4  256     large:  12 8 512
    # arch = small

    pretraining = False
    pretraining_str = 'pretraining' if pretraining else ''

    trained_epoch = 10
    # trained_epoch = 9

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
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0)

    if pretraining:
        # temp = Drug_BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)

        temp.load_weights("./Result/Pre/bert_weightss_ab_128_data_s2_10.h5")

        # temp.load_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],trained_epoch))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder{}_{}.h5'.format(arch['name'],trained_epoch))
        print('load_wieghts')


    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) 

    auc= -10
    stopping_monitor = 0
    for epoch in range(250):
        accuracy_object = tf.keras.metrics.BinaryAccuracy()
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
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

        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)
        print(y_true)
        print(y_preds)

        val_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
        print('val auc:{:.4f}'.format(auc_new), 'val accuracy:{:.4f}'.format(val_accuracy))

        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0
            np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch,pretraining_str),
                    [y_true, y_preds])
            model.save_weights('classification_weights/{}_nopre_{}.h5'.format(task,seed))
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
    model.load_weights('classification_weights/{}_nopre_{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    y_preds = tf.sigmoid(y_preds).numpy()
    test_auc = roc_auc_score(y_true, y_preds)
    test_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
    print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_accuracy))

    return test_auc,test_accuracy

if __name__ == '__main__':

    auc_list = []
    acc_list = []
    for seed in [121,225,666]: # 121    508 s2的
        print(seed)
        auc,acc = main(seed)
        auc_list.append(auc)
        acc_list.append(acc)
    print(auc_list)
    print(acc_list)
