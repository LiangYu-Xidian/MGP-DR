import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================================================================================

# MultiHeadAttention
#    ↑
# EncoderLayer
#    ↑
# Encoder
#    ↑
# BertModel / PredictModel

# ======================================================================  函  数  ==============================================================

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


def scaled_dot_product_attention(q, k, v, mask,adjoin_matrix):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)  # 生成 q*k 矩阵相乘

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor. 为了将Q和K中一些填充的为零的向量清除掉,因为在softmax之前将相对应的位置设置为-1e10，这么小的数在softmax之后就会变为0，起到mask的作用
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # （0是想要的很小的数是不想要的）把原本邻接矩阵中0的位置变成一个很小的数，在softmax后就会趋向于0
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
      return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=gelu),  # (batch_size, seq_len, dff)tf.keras.layers.LeakyReLU(0.01)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


# ================================================================ Encoder ==========================================================================



class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads  # 整数除法，返回不大于结果的一个最大整数

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask,adjoin_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,adjoin_matrix)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights






class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        attn_output, attention_weights = self.mha(x, x, x, mask,adjoin_matrix)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2,attention_weights






class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        
        print(seq_len)
        print(adjoin_matrix.shape[0])
        print(adjoin_matrix.shape)

        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:] # tf.newaxis的主要用途是增加一个维度
        
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        attention_weights_list = []

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
        
            attention_weights_list.append(attention_weights)

        return x,  # (batch_size, input_seq_len, d_model)




class Encoder_test(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder_test, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        attention_weights_list = []
        xs = []

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
            attention_weights_list.append(attention_weights)
            xs.append(x)

        return x,attention_weights_list,xs





# ============================================================= BertModel =============================================================================



class BertModel(tf.keras.Model):
    def __init__(self,num_layers = 6, d_model = 256, dff = 512, num_heads = 4, vocab_size = 17, dropout_rate = 0.1):  # dff = FFN size 是 2倍的d_model
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu) 
        self.layernorm = tf.keras.layers.LayerNormalization(-1)  # 计算每一条记录的均值和标准差，再进行标准化
        self.fc2 = tf.keras.layers.Dense(vocab_size) # 线性激活

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)  # training 是 True or False
        x = self.fc1(x)  # 输出 embedding size 为 256 （d_model） 维的向量
        x = self.layernorm(x)
        x = self.fc2(x)  # 输出 embedding size 为 17 维的向量，分别对应词典中的17个元素
        return x




class BertModel_test(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 17,dropout_rate = 0.1): 
        super(BertModel_test, self).__init__()
        self.encoder = Encoder_test(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
    def call(self,x,adjoin_matrix,mask,training=False):
        x,att,xs = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)  
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x,att,xs




class Drug_BertModel(tf.keras.Model):
    def __init__(self,num_layers = 6, d_model = 256, dff = 512, num_heads = 4, vocab_size = 17, dropout_rate = 0.1):  # dff = FFN size 是 2倍的d_model
        super(Drug_BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,rate=dropout_rate)

        # 预训练策略一：遮盖
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu) 
        self.layernorm = tf.keras.layers.LayerNormalization(-1)  # 计算每一条记录的均值和标准差，再进行标准化
        self.fc2 = tf.keras.layers.Dense(vocab_size) # 线性激活

        #预训练策略二：靶标信息
        self.fc3 = tf.keras.layers.Dense(d_model,activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.fc4 = tf.keras.layers.Dense(1)

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)  # training 是 True or False

        x1 = self.fc1(x)  # 输出 embedding size 为 d_model 维的向量
        x1 = self.layernorm(x1)
        x1 = self.fc2(x1)  # 输出 embedding size 为 17 维的向量，分别对应词典中的17个元素

        x2 = x[:,0,:] # 拿出分子整体的embedding
        x2 = self.fc3(x2)
        x2 = self.dropout(x2,training=training)
        x2 = self.fc4(x2) # 输出一维向量
        return x1,x2


# ============================================================ PredictModel =========================================================================



class PredictModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size =17,dropout_rate = 0.1,dense_dropout=0.1):
        super(PredictModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(d_model,activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = x[:,0,:]
        x = self.fc1(x)
        x = self.dropout(x,training=training)
        x = self.fc2(x) # 输出 embedding size 为 1 维的向量，即预测概率（类标签）
        return x



class PredictModel_test(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size =17,dropout_rate = 0.1,dense_dropout=0.5):
        super(PredictModel_test, self).__init__()
        self.encoder = Encoder_test(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(d_model, activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self,x,adjoin_matrix,mask,training=False):
        x,att,xs = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        SUP_embedding = x
        x = self.fc2(x) 
        return x,SUP_embedding




# ============================================================ Multi_PredictModel =========================================================================


class Multi_PredictModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size =17,dropout_rate = 0.1,dense_dropout=0.1):
        super(Multi_PredictModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(d_model,activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(86)  # 有86类

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = x[:,0,:]
        x = self.fc1(x)
        x = self.dropout(x,training=training)
        x = self.fc2(x) # 输出 embedding size 为 86 维的向量，即预测概率（类标签）
        return x


