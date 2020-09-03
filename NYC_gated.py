# -*- coding:  UTF-8 -*-
from __future__ import division
import tensorflow as tf
import numpy as np
from compiler.ast import flatten
from AMSGrad import AMSGrad
from sklearn.metrics import roc_auc_score
import math
import matplotlib.pyplot as plt
import os
import sklearn.metrics as metric

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
#NYC for WWW compared model (Next POI Recommendation)
import time
# build dictionary
voc_poi = list() #pois in dictionary including <PAD>, <GO> and <EOS>
table_X = {}
new_table_X={}
def getXs():  # 读取轨迹向量
    fpointvec = open('traindata/NYC_vector_256.dat', 'r')  # 获取check-in向量 已经用word2vec训练得到 tor traindata/
    #     table_X={}  #建立字典索引
    item = 0
    for line in fpointvec.readlines():
        lineArr = line.split()

        if (len(lineArr) < 256):  # delete fist row
            continue
        item += 1  # 统计条目数
        X = list()
        for i in lineArr[1:]:
            X.append(float(i))  # 读取向量数据
        if lineArr[0] == '</s>':
            table_X['<PAD>'] = X  # dictionary is a string  it is not a int type
        else:
            table_X[lineArr[0]] = X
    print 'finish read vector of POIs'
    return table_X
def extract_words_vocab():
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch]) #取最大长度
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
#Read data
def read_pois():
    Train_DATA = []
    Train_USER = []
    Test_DATA = []
    Test_USER = []
    T_DATA={}
    fread_train=open('traindata/NYC_train.dat', 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Train_DATA.append(data_line)
        Train_USER.append(line[0])
        T_DATA.setdefault(line[0],[]).append(data_line)


    fread_train=open('traindata/NYC_test.dat', 'r')
    for lines in fread_train.readlines():
        line=lines.split()
        data_line = list()
        for i in line[1:]:
            data_line.append(i)
        Test_DATA.append(data_line)
        Test_USER.append(line[0])
    print 'Train Size', len(Train_DATA)
    print 'total trajectory',len(Test_DATA)+len(Train_DATA)
    return T_DATA,Train_DATA, Train_USER, Test_DATA, Test_USER


#read pois
getXs() #read vectors
H_DATA,Train_DATA, Train_USER, Test_DATA, Test_USER=read_pois()
#print H_DATA['1']
#print History['1']
T=Train_DATA+Test_DATA
total_check=len(flatten(T))
total_user=set(flatten(Train_USER+Test_USER))
user_number=len(set(total_user))
for i_ in range(len(T)):
    for j_ in range(len(T[i_])):
        new_table_X[T[i_][j_]]=table_X[T[i_][j_]] #new pois dictionary for dataset

#add additional characters
new_table_X['<GO>'] = table_X['<GO>']
new_table_X['<EOS>'] = table_X['<EOS>']
new_table_X['<PAD>'] = table_X['<PAD>']
for poi in new_table_X:
    voc_poi.append(poi)
print 'lens',len(voc_poi)
int_to_vocab, vocab_to_int=extract_words_vocab()
print 'Dictionary Length', len(int_to_vocab),'POI number',len(int_to_vocab)-3
TOTAL_POI=len(int_to_vocab)
print 'Total check-ins',total_check,TOTAL_POI
print 'Total Users',user_number

History={}
for key in H_DATA.keys(): # index char
    temp=H_DATA[key]
    temp=flatten(temp)
    new_temp=[]
    for i in temp:
        new_temp.append(vocab_to_int[i])
        #print vocab_to_int[i]
    History[key]=new_temp
#convert data
new_trainT = list()
for i in range(len(Train_DATA)): #TRAIN
    temp = list()
    for j in range(len(Train_DATA[i])):
        temp.append(vocab_to_int[Train_DATA[i][j]])
    new_trainT.append(temp)
max_check=max(flatten(new_trainT))
print max_check

new_testT = list()
for i in range(len(Test_DATA)):
    temp = list()
    for j in range(len(Test_DATA[i])):
        temp.append(vocab_to_int[Test_DATA[i][j]])
    new_testT.append(temp)

#Creat dictionary embeddings
def dic_em():
    dic_embeddings=list()
    for key in new_table_X:
        dic_embeddings.append(new_table_X[key])
    return dic_embeddings
dic_embeddings=tf.constant(dic_em())
#initial Deep Learning
n_hidden = 300
en_n_hidden=300
batch_size =16 # batch
keep_prob = tf.placeholder("float")
it_learning_rate = tf.placeholder("float")
train_iters =30# 遍历样本次数 for training
attention_size=300
z_size=128

#placeholder
input_x = tf.placeholder(dtype=tf.int32)
next_x= tf.placeholder(dtype=tf.float32, shape=[batch_size,TOTAL_POI])
encoder_embed_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
history_input = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
def get_onehot(index):
    x = [0] * TOTAL_POI
    x[index] = 1
    return x
#W B
with tf.name_scope("w_b"):
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden+en_n_hidden,TOTAL_POI]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([TOTAL_POI]))
    }
#encoder layer
def get_encoder_layer(encoder_input,keep_prob,reuse=False):
    with tf.variable_scope("encoder",reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_input)
        input_=tf.transpose(encoder_input,[1,0,2])
        encode_gru = tf.contrib.rnn.GRUCell(en_n_hidden)
        encode_cell = tf.contrib.rnn.DropoutWrapper(encode_gru, output_keep_prob=keep_prob)
        (outputs, states) = tf.nn.dynamic_rnn(encode_cell, input_, time_major=True, dtype=tf.float32)

        o_mean = tf.contrib.layers.fully_connected(inputs=states, num_outputs=z_size, activation_fn=None,
                                                    scope="z_mean")  # relu tf.nn.sigmoid
        o_stddev = tf.contrib.layers.fully_connected(inputs=states, num_outputs=z_size, activation_fn=None,
                                                      scope="z_std")  # relu activation_fn=tf.nn.sigmoid
        return outputs, states,o_mean,o_stddev

# encoder layer
w_omegas= tf.Variable(tf.random_normal([256, n_hidden], stddev=0.1))
b_omegas = tf.Variable(tf.random_normal([n_hidden], stddev=0.1))

vw_omegas= tf.Variable(tf.random_normal([256, 2*n_hidden], stddev=0.1))
vb_omegas = tf.Variable(tf.random_normal([2*n_hidden], stddev=0.1))

w_omega= tf.Variable(tf.random_normal([n_hidden, attention_size], stddev=0.1))
b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), dtype=tf.float32)

in_dim=300
out_dim=600
kernel_size=5
def gated_linear_units(inputs):
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 3
  input_pass = inputs[:,:,0:int(input_shape[2]/2)]
  input_gate = inputs[:,:,int(input_shape[2]/2):]
  input_gate = tf.nn.relu(input_gate)
  return tf.multiply(input_pass, input_gate)
def get_history_layer(encoder_input, keep_prob,ht, reuse=False):
    with tf.variable_scope("history", reuse=reuse):
        encoder_input = tf.nn.embedding_lookup(dic_embeddings, encoder_input)
        input_ = tf.transpose(encoder_input, [0, 1, 2])

        out=tf.nn.relu(tf.tensordot(input_,w_omegas, axes=1)+b_omegas) #tf.nn.tanh(
        V = tf.get_variable('Vs', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0,
                                                                     stddev=tf.sqrt(
                                                                         4.0 * 1.0/ (kernel_size * in_dim))),
                            trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])  # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('gs', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('bs', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
        inputs = tf.contrib.layers.dropout(
            inputs=input_,
            keep_prob=1.0)
        # inputs2=tf.transpose(inputs2,[0,2,1])

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0, 1])
        #print 'W',tf.shape(W)
        out = tf.nn.bias_add(tf.nn.conv1d(out, W, stride=1, padding="SAME"), b)
        out=gated_linear_units(out)
        # A=tf.tensordot(input_,w_omegas, axes=1)+b_omegas
        # #print 'A', A
        # B=tf.nn.sigmoid(tf.tensordot(input_,vw_omegas, axes=1)+vb_omegas) #
        #
        # out=tf.multiply(A,B)


        inputs = tf.transpose(out, [1, 0, 2])
        v = tf.tanh(ht*(tf.tensordot(inputs, w_omega, axes=1) + b_omega))
        v = tf.transpose(v, [1, 0, 2])
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
        output = tf.reduce_sum(out * tf.expand_dims(alphas, -1), 1)
        # attention_output, alphas = attentions(out,attention_size, ht,time_major=True,
        #                                        return_alphas=True)  # give a weight to each timestep
        drop = tf.nn.dropout(output, keep_prob)
        #print 'drop', drop

        return drop

#compute
encode_outputs, encode_states,z_mean,z_stddev= get_encoder_layer(encoder_embed_input,keep_prob)
I=encode_outputs[-1]#[-1]

samples=tf.random_normal(tf.shape(z_stddev))
z=z_mean+tf.exp(z_stddev*0.5)*samples
fw_w_mean=tf.Variable(tf.random_normal([z_size, n_hidden]))
fw_b_mean=tf.Variable(tf.random_normal([n_hidden]))

state_fw=tf.nn.relu(tf.matmul(z,fw_w_mean)+fw_b_mean)


history=get_history_layer(history_input, keep_prob,ht=state_fw)

context=tf.concat([I,history],axis=1)
print 'ss',context
pred= tf.matmul(context, weights['out']) + biases['out']
preds=tf.nn.softmax(pred)
#optimizer
latent_loss = 0.5 * tf.reduce_sum(tf.exp(z_stddev) - 1. - z_stddev + tf.square(z_mean), 1)
latent_loss=tf.reduce_mean(latent_loss)
cost= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=next_x, logits=pred))  #
T_COST=cost+0.01*latent_loss
optimizer_class = AMSGrad(learning_rate=it_learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8).minimize(T_COST)
pred_mask = tf.arg_max(pred,1)  # a list of result

def eos_sentence_batch(sentence_batch,eos_in):
    return [sentence+[eos_in] for sentence in sentence_batch]

initial = tf.global_variables_initializer()

def  compute():
    print 'start train'
    all_vars = tf.trainable_variables()
    for v in all_vars[9:17]:
        print 'name:', v.name
    saver = tf.train.Saver(all_vars[9:17])  # variables
    with tf.Session() as sess:
        sess.run(initial)
        saver.restore(sess, './logs/NYC_cnn_gated_new2.pkt')
        # merged = tf.summary.merge_all()  #
        # writer = tf.summary.FileWriter('./logs/', sess.graph)

        #sort original data
        index_T = {}
        trainT = []
        trainU = []
        for i in range(len(new_trainT)):
            index_T[i] = len(new_trainT[i])
        temp_size = sorted(index_T.items(), key=lambda item: item[1])
        for i in range(len(temp_size)):
            id = temp_size[i][0]
            trainT.append(new_trainT[id])
            trainU.append(Train_USER[id])
        # sort for test dataset
        index_T = {}
        testT = []
        testU = []
        for i in range(len(new_testT)):
            index_T[i] = len(new_testT[i])
        temp_size = sorted(index_T.items(), key=lambda item: item[1])
        for i in range(len(temp_size)):
            id = temp_size[i][0]
            testT.append(new_testT[id])
            testU.append(Test_USER[id])
        # -----------------------------------
        #epoch
        train_size=len(trainT)%batch_size
        test_size=len(testT)%batch_size
        trainT=trainT+trainT[-(batch_size-train_size):] #copy data and fill the last batch size
        trainU=trainU+trainU[-(batch_size-train_size):]

        testT=testT+testT[-(batch_size-test_size):] #copy data and fill the last batch size
        testU=testU+testU[-(batch_size-test_size):]
        learning_rate = 0.001  # learning rate
        epoch = 0
        Flag = True
        while (Flag):
            epoch += 1
            if (epoch == (train_iters) and (learning_rate is not 0.0002)):
                epoch = 0
                learning_rate = learning_rate - 0.0002
            if (learning_rate < 0.0002):
                Flag = False
                break  # end computing
            print 'learning rate', learning_rate
            step=0
            epoch_loss=0
            temp_loss=0
            tempz_loss=0
            Train_Acc_1=0
            Train_Acc_5= 0
            temp_acc=0
            while step < len(trainU)// batch_size:
                start_i = step * batch_size
                Label_x = []
                batch_x =[]
                embedding_x=[]
                #for test
                input_x = trainT[start_i:start_i + batch_size]
                History_batch=[]
                input_y=trainU[start_i:start_i + batch_size]
                for uid in input_y:
                    History_batch.append(History[uid])
                history_x=pad_sentence_batch(History_batch,vocab_to_int['<PAD>'])
                #print history_x
                #print 'input_x',input_x
                for i in input_x:
                    Label_x.append(i[-1]) #add label in each batch size
                    batch_x.append(i[:-1]) #add n-1 pois
                    embedding_x.append(get_onehot(i[-1]))
                sources_batch = pad_sentence_batch(batch_x, vocab_to_int['<PAD>'])
                # 记录长度
                pad_source_lengths = []
                for source in batch_x:
                    pad_source_lengths.append(len(source) + 1)
                z_loss,pred_poi,loss,optimizer=sess.run([latent_loss,pred,cost,optimizer_class],feed_dict={history_input:history_x,encoder_embed_input:sources_batch,next_x:embedding_x,it_learning_rate:learning_rate,keep_prob:0.5})
                epoch_loss+=loss  #
                temp_loss+=loss
                tempz_loss+=z_loss

                for i in range(batch_size):
                    value = pred_poi[i]
                    top1 = np.argpartition(a=-value, kth=1)[:1]
                    top5 = np.argpartition(a=-value, kth=5)[:5]
                    if top1==input_x[i][-1]:
                        Train_Acc_1 += 1
                        temp_acc += 1
                    if input_x[i][-1] in top5:
                        Train_Acc_5+=1
                if(step%1000==0 and step!=0):
                    print step,temp_acc
                    print 'temp loss',temp_loss,'acc of each',temp_acc/(1000*batch_size)
                    temp_acc=0
                    temp_loss=0

                step += 1
            print 'epoch is',epoch
            print 'train item',step,tempz_loss,epoch_loss
            print 'train accuracy@1: ', Train_Acc_1 / (step*batch_size),Train_Acc_1/len(trainU)
            print 'train accuracy@5: ', Train_Acc_5 / (step * batch_size), Train_Acc_5 / len(trainU)
            test(testT,testU,sess)
            saver.save(sess, './logs/NYC_cnn_gated_new2.pkt')

def test(testT,testU,sess):
    step = 0
    Test_Acc_1 = 0
    Test_Acc_5 = 0
    Test_Acc_10 = 0
    Prob_Y=[]
    Y=[]
    y=[]
    p_y=[]
    while step < len(testU) // batch_size:
        Label_x = []
        batch_x = []
        embedding_x = []
        start_i = step * batch_size
        # for test
        input_x = testT[start_i:start_i + batch_size]
        History_batch = []
        input_y = testU[start_i:start_i + batch_size]
        for uid in input_y:
            History_batch.append(History[uid])
        history_x = pad_sentence_batch(History_batch, vocab_to_int['<PAD>'])
        # print 'input_x',input_x
        for i in input_x:
            Label_x.append(i[-1])  # add label in each batch size
            batch_x.append(i[:-1])  # add n-1 pois
            embedding_x.append(get_onehot(i[-1]))
        sources_batch = pad_sentence_batch(batch_x, vocab_to_int['<PAD>'])
        # 记录长度
        pad_source_lengths = []
        for source in batch_x:
            pad_source_lengths.append(len(source) + 1)
        pred_poi = sess.run(preds,feed_dict={history_input:history_x ,encoder_embed_input: sources_batch, next_x: embedding_x,keep_prob: 1.0})
        for i in range(batch_size):
            value = pred_poi[i]
            true_value = get_onehot(input_x[i][-1])
            top1 = np.argpartition(a=-value, kth=1)[:1]
            top5=np.argpartition(a=-value, kth=5)[:5]
            top10 = np.argpartition(a=-value, kth=10)[:10]
            Y.append(true_value)
            Prob_Y.append(value)
            y.append(input_x[i][-1])
            p_y.append(top1)
            #print top1
            #print input_x[i][-1]
            if top1 == input_x[i][-1]:
                Test_Acc_1+= 1
            if input_x[i][-1]in top5:
                Test_Acc_5+=1
            if input_x[i][-1]in top10:
                Test_Acc_10+=1
        step += 1
    print '\n --Test accuracy@1: ', Test_Acc_1 / (step*batch_size),Test_Acc_1/len(testU)
    print '\n --Test accuracy@5: ', Test_Acc_5 / (step * batch_size), Test_Acc_5 / len(testU)
    print '\n --Test accuracy@10: ', Test_Acc_10 / (step * batch_size), Test_Acc_10 / len(testU)
    print len(Y)
    print len(Prob_Y)

    Y=np.array(Y)
    Prob_Y=np.array(Prob_Y)
    print 'auc_value', metric.roc_auc_score(Y.T, Prob_Y.T, average='micro')  # ,,average='micro'
    print 'MAP_value', metric.average_precision_score(Y.T, Prob_Y.T, average='micro')  # ,,average='micro'
    print '----\n'
    print '----\n'
print "time.time(): %f ",time.time()
time_s=time.time()
compute()
time_e=time.time()
print 'NYC time consuming(s):',(time_e-time_s)