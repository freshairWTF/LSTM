# -*- coding: utf-8 -*-   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as skflow
import tensorflow.contrib.keras as K
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import os


input_size = 10
output_size = 1
lr = 1e-3        
layers_num = 3
time_step = 22
batch_size = 30
epochs = 100
L2_beta = 0.00001
epsilon = 0.001
early_stop_patience = 10

def load_data(file_name, sequence_length = 55, split = 0.99):
    df = pd.read_csv(file_name)
    data_all = Date_load[['vol', 'open', 'upper', 'lower', 'MA:MA1', 'MA:MA2', 'MA:MA3', 'MA:MA4', 'RSI:RSI', 'RSI:buy', 'RSI:sell', 'BOLL:UpLine', 'BOLL:DownLine', 'BOLL:MidLine']]
    data_all = np.array(df).astype(float)
    #minmaxscaler
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    #sequence
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    #np.random.shuffle(reshaped_data)
    #
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]
    train_y = y[: split_boundary]
    test_y = y[split_boundary:]
    return train_x, train_y, test_x, test_y, scaler

def deal_data(filename, sequence_length, cut_length):
    #read data
    Date_load = pd.read_csv(filename)
    X = Date_load[['vol', 'open', 'upper', 'lower', 'MA:MA1', 'MA:MA2', 'MA:MA3', 'MA:MA4', 'RSI:RSI',  'BOLL:MidLine']]
    Y = Date_load['last']
    X = np.array(X)
    Y = np.array(Y)
    data_x = []
    data_y = []    
    #split
    cut_num = len(X) * cut_length // 100
    x_train = np.array(X[:cut_num] )
    x_test = np.array(X[cut_num:])
    y_train = np.array(Y[:cut_num])
    y_test = np.array(Y[cut_num:]) 
    print(cut_num)
    #minmaxscaler
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    y_train = scaler.fit_transform(y_train) 
    x_test = scaler.fit_transform(x_test)
    y_test = scaler.fit_transform(y_test)
    #scaler=StandardScaler()
    #x_train=scaler.fit_transform(x_train)
    #x_test=scaler.transform(x_test)    
    #y_train = scaler.fit_transform(y_train)
    #val data
    np.random.shuffle(x_train)
    np.random.shuffle(y_train)
    x_val = np.array(x_train[: len(x_test)])
    y_val = np.array(y_train[: len(x_test)]) 
    #sequence
    def sequence_data(data, sequence_length):
        data_x = []
        for i in range(len(data) - sequence_length):
            temp = [data[i+j] for j in range(sequence_length)]
            data_x.append(temp)
        data_x = np.array(data_x).astype('float32')
        return data_x
    x_train = sequence_data(x_train, sequence_length)
    x_test = sequence_data(x_test, sequence_length)
    y_train = sequence_data(y_train, sequence_length)
    y_test = sequence_data(y_test, sequence_length)
    x_val = sequence_data(x_val, sequence_length)
    y_val = sequence_data(y_val, sequence_length)
    #dim up
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    #return
    return x_train, x_test, y_train, y_test, x_val, y_val, scaler

def batch_norm_wrapper(inputs, is_training, layer_name, decay = 0.97):
    layer_name = 'batch_norm%s' % layer_name
    with tf.name_scope(layer_name):
        with tf.name_scope('scale'):
            scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
            tf.summary.histogram(layer_name + 'scale', scale) 
        with tf.name_scope('beta'):
            beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
            tf.summary.histogram(layer_name + 'beta', beta) 
        with tf.name_scope('pop_mean'):
            pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
            tf.summary.histogram(layer_name + 'pop_mean', pop_mean) 
        with tf.name_scope('pop_var'):
            pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
            tf.summary.histogram(layer_name + 'pop_var', pop_var) 
        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                bn = tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
        else:
            bn = tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
        tf.summary.histogram(layer_name + 'bn', bn) 
        return bn
def Dense(inputs, in_size, out_size, layer_name):
    regularizer = tf.contrib.layers.l1_regularizer(scale = 0.01)
    layer_name = 'Dense%s' % layer_name
    with tf.name_scope(layer_name):
        with tf.variable_scope(name_or_scope = 'weights', regularizer = regularizer):
            Weights = tf.Variable(tf.truncated_normal([in_size,out_size]), name = 'W')
            tf.summary.histogram(layer_name + 'weights', Weights) 
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + 'biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            outputs = Wx_plus_b
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.summary.histogram(layer_name + '/regularization_loss', regularization_loss)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs, regularization_loss    

def lstm(X, is_training , keep_prob):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    input0 = tf.reshape(X,[-1, input_size])       #3D--2D
    input_rnn0, reg_loss0 = Dense(input0, input_size, 1024, layer_name = 1)
    input_bn0 = batch_norm_wrapper(input_rnn0, is_training, layer_name = 1)
    input_ac0 = tf.nn.relu(input_bn0)
    if is_training and keep_prob < 1.0 :
        input_ac0 = tf.nn.dropout(input_ac0, 0.5)
    input_rnn1, reg_loss1 = Dense(input_ac0, 1024, 256, layer_name = 2)
    input_bn1 = batch_norm_wrapper(input_rnn1, is_training, layer_name = 2)
    inpu_ac1 = tf.nn.relu(input_bn1)
    if is_training and keep_prob < 1.0 :
        inpu_ac1 = tf.nn.dropout(inpu_ac1, 0.5)
    input_rnn = tf.reshape(inpu_ac1, [-1, time_step, 256])                  #2D--3D
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units = 256, forget_bias = 1.0)
    if is_training and keep_prob < 1.0 :
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layers_num, state_is_tuple = True)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn,final_states = tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, time_major = False, dtype=tf.float32)
    print(np.shape(output_rnn))
    output0 = tf.reshape(output_rnn, [-1, 256]) 
    output_rnn0, reg_loss2 = Dense(output0, 256, 512, layer_name = 3)
    output_bn0 = batch_norm_wrapper(output_rnn0, is_training, layer_name = 3)
    output_ac0 = tf.nn.relu(output_bn0)
    if is_training and keep_prob < 1.0 :
        output_ac0 = tf.nn.dropout(output_ac0, 0.5)
    pred0, reg_loss3 = Dense(output_ac0, 512, 1, layer_name = 4)
    pred = batch_norm_wrapper(pred0, is_training, layer_name = 4)
    reg_loss = tf.reduce_sum(reg_loss0 + reg_loss1 + reg_loss2 + reg_loss3)
    #pred = tf.nn.sigmoid(pred0)
    tf.summary.histogram('pred', pred)
    return pred, reg_loss

def train_lstm(batch_size, time_step):
    pred, reg_loss = lstm(X, is_training = True, keep_prob = 0.5)
    y_reshape = tf.reshape(Y, [-1, output_size])
    #regularizer
    # regularizer_l1 = tf.contrib.layers.l1_regularizer(scale)
    # regularizer_l2 = tf.contrib.layers.l2_regularizer()
    #loss fuc
    with tf.name_scope('loss'):
        global_step = tf.Variable(0, trainable = False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 2000, 0.5, staircase = True)    #learning_decay
        #L2_loss = L2_beta * (tf.nn.l2_loss(weights['in1']) + tf.nn.l2_loss(weights['in2']) + tf.nn.l2_loss(weights['out1']) + tf.nn.l2_loss(weights['out2'])
                             #+ tf.nn.l2_loss(biases['in1']) + tf.nn.l2_loss(biases['in2']) + tf.nn.l2_loss(biases['out1']) + tf.nn.l2_loss(biases['out2']))
        loss = tf.reduce_mean((tf.square(pred - y_reshape)) + reg_loss)
        val_loss = tf.reduce_mean(tf.square(pred - y_reshape)) 
        #val_loss = tf.contrib.metrics.accuracy(pred, y_val)
        #loss = tf.reduce_sum(loss)
        #loss=tf.reduce_mean(tf.square(pred[-1] - y_reshape[-1]))
        # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels = y_reshape, logits = pred))
        tf.summary.scalar('regulazation_loss', reg_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('val_loss', val_loss)
    #train
    with tf.name_scope('train_op'):
        train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
    saver = tf.train.Saver()
    best_validation_loss = float("inf")
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("C:/Users/Administrator/logs", sess.graph)
        #test_writer = tf.summary.FileWriter("C:/Users/Administrator/logs", sess.graph)   
        sess.run(tf.global_variables_initializer())
        save_path = saver.save(sess, "d:/variables/mulitiLSTM1.ckpt")
        for i in range(epochs):
            np.random.shuffle(x_train)
            for step in range(len(x_train)//batch_size-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:x_train[step*batch_size:(step+1)*batch_size],Y:y_train[step*batch_size:(step+1)*batch_size]} )
                print('epochs:%s, batch_size_step:%s, train_loss:%s' %(i, step, loss_))
            for step in range(len(x_val)//batch_size-1):
                _,val_loss_ = sess.run([train_op, val_loss], feed_dict={X:x_val[step*batch_size:(step+1)*batch_size],Y:y_val[step*batch_size:(step+1)*batch_size]})
                print('val_loss:%s' %val_loss_)
                if val_loss_ < best_validation_loss:
                    best_validation_loss = val_loss_
                    current_epoch = 0
                    print("model saved: %s, loss: %s" %(save_path, val_loss_))
                else:
                    current_epoch = current_epoch + 1
                    print('early_stopping_step:%s' %current_epoch)
                if current_epoch >= early_stop_patience :
                    print('early stopping')
                    return 
            if i % 1== 0:
                train_result = sess.run(merged,  feed_dict={X:x_train[step*batch_size:(step+1)*batch_size],Y:y_train[step*batch_size:(step+1)*batch_size]})
                train_writer.add_summary(train_result, i)
                print("model saved: %s, loss: %s" %(save_path, loss_))
                print(i,loss_)
                              
def _test():
        x_train, x_test, y_train, y_test, x_val, y_val, scaler=deal_data('e:/rb_data.csv',22, 95)
        y_reshape = tf.reshape(Y, [-1, output_size])
        pred,_=lstm(X, is_training = False, keep_prob = 1.0)
        loss = tf.reduce_mean(tf.square(pred - y_reshape)) 
        saver=tf.train.Saver()
        with tf.Session() as sess:
            #restore
            os.path.exists('d:/variables/mulitiLSTM1.ckpt')
            saver.restore(sess, 'd:/variables/mulitiLSTM1.ckpt')
            #test
            test_predict=[]
            for step in range(len(x_test)):
                pred_, loss_=sess.run([pred, loss] ,feed_dict={X:[x_test[step]], Y: [y_test[step]]})
                print('test_loss:%s' %loss_)
                prob = pred_[-1]
                test_predict.append(prob)
            total_loss = np.mean(loss_ )
            print('test_loss mean:%s' %total_loss)            
            y_test = y_test[:,-1]
            test_predict = scaler.fit_transform(test_predict)
            plt.plot(test_predict,color='b')
            plt.plot(y_test, color='r')
            plt.show()
            y_test = scaler.inverse_transform(y_test)
            test_predict = scaler.inverse_transform(test_predict)
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b')
            plt.plot(list(range(len(y_test))), y_test,  color='r')
            plt.show()
            
if __name__=='__main__':
    with tf.name_scope('inputs'):
        X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
        Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
        keep_prob = tf.placeholder(tf.float32)
    x_train, x_test, y_train, y_test, x_val, y_val, scaler=deal_data('e:/rb_data.csv',22, 95)
    train_lstm(batch_size, time_step)
    #_test()
