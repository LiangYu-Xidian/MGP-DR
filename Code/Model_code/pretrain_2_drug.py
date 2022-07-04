import tensorflow as tf
from model import  Drug_BertModel
from Dataset_2 import Drug_Bert_Dataset
import time
import os
from tensorflow import keras as keras


keras.backend.clear_session()

optimizer = tf.keras.optimizers.Adam(1e-4) #1e-4

small = {'name': 'Small', 'num_layers': 3, 'num_heads': 2, 'd_model': 64, 'path': './small_weights','addH':False}
medium = {'name': 'Medium', 'num_layers': 3, 'num_heads': 2, 'd_model': 256, 'path': 'medium_weights','addH':True}
large = {'name': 'Large_128_s_ab_r2_s1', 'num_layers': 3, 'num_heads': 2, 'd_model': 128, 'path': 'large_weights','addH':True}
medium_without_H = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'weights_without_H','addH':False}

arch = large      ## small 3 2 128   medium: 6 4  256     large:  12 8 512

num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']
addH = arch['addH']
dff = d_model*2
vocab_size =17
dropout_rate = 0.1

model = Drug_BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)

# train_dataset, test_dataset = Graph_Bert_Dataset(path='D:/TASK2/Data/Drugbank/SMILE/filtered_approved.csv',smiles_field='SMILES',addH=addH).get_data()
train_dataset, test_dataset = Drug_Bert_Dataset(path='./S_AB.csv',smiles_field='SMILES',label_field='S_AB',addH=addH).get_data()

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')


train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
train_mse = tf.keras.metrics.MeanSquaredError(name='train_mse')
test_mse = tf.keras.metrics.MeanSquaredError(name='test_mse')

loss_function_1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_function_2 = tf.keras.losses.MeanSquaredError()



def train_step(x, adjoin_matrix, y, char_weight,s_ab):
    # print(x.get_shape())
    # print(adjoin_matrix.get_shape())
    # print(y.get_shape())
    # print(char_weight.get_shape())
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    # print(seq.get_shape())
    mask = seq[:, tf.newaxis, tf.newaxis, :]  
    # print(mask.get_shape())
    with tf.GradientTape() as tape:
        predictions_1,predictions_2 = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=True)
        loss_1 = loss_function_1(y,predictions_1,sample_weight=char_weight)
        print("loss_1:",loss_1)
        loss_2 = loss_function_2(s_ab,predictions_2)
        print("loss_2:",loss_2)
        loss = loss_1 + loss_2
        print("loss:",loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(y,predictions_1,sample_weight=char_weight)
    train_mse.update_state(s_ab,predictions_2)


@tf.function(input_signature=train_step_signature)
def test_step(x, adjoin_matrix, y, char_weight,s_ab):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    predictions_1,predictions_2 = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=False)
    test_accuracy.update_state(y,predictions_1,sample_weight=char_weight)
    test_mse.update_state(s_ab,predictions_2)


for epoch in range(10):
    start = time.time()
    train_loss.reset_states()

    for (batch, (x, adjoin_matrix , y , char_weight, s_ab)) in enumerate(train_dataset):
        train_step(x, adjoin_matrix, y , char_weight, s_ab)

        if batch % 500 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.result()))
            print('Accuracy: {:.4f}'.format(train_accuracy.result()))
            print('MSE: {:.4f}'.format(train_mse.result()))
            
            for x, adjoin_matrix ,y , char_weight, s_ab in test_dataset:
                test_step(x, adjoin_matrix, y , char_weight, s_ab)
            print('Test Accuracy: {:.4f}'.format(test_accuracy.result()))
            print('Test MSE: {:.4f}'.format(test_mse.result()))
            test_accuracy.reset_states()
            train_accuracy.reset_states()
            test_mse.reset_states()
            train_mse.reset_states()

    print(arch['path'] + '/bert_weights{}_{}.h5'.format(arch['name'], epoch+1))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Accuracy: {:.4f}'.format(train_accuracy.result()))
    print('MSE: {:.4f}'.format(train_mse.result()))
    
    model.save_weights(arch['path']+'/bert_weights{}_{}.h5'.format(arch['name'],epoch+1))
    print('Saving checkpoint')

for x, adjoin_matrix ,y , char_weight, s_ab in test_dataset:
        test_step(x, adjoin_matrix, y , char_weight, s_ab)
print('Test Accuracy: {:.4f}'.format(test_accuracy.result()))
print('Test MSE: {:.4f}'.format(test_mse.result()))
# test_accuracy.reset_states()

