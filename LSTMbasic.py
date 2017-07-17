#coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import os

#定义常量
rnn_unit=10       #hidden layer units
input_size=4
output_size=1
lr=0.0005         #学习率
#——————————————————导入数据——————————————————————


def deal_data(filename,sequence_length):
    #数据读取
    Date_load = pd.read_csv(filename)
    X = Date_load[['vlo', 'open', 'upper', 'lower']]
    Y = Date_load['last']
    scaler = MinMaxScaler()
    X_trans = scaler.fit_transform(X)
    y_trans = scaler.fit_transform(Y)
    data_x = []
    data_y = []
    #归一化、序列化
    for i in range(len(X_trans) - sequence_length):
        temp = [X_trans[i+j] for j in range(sequence_length)]
        data_x.append(temp)
        reshaped_X = np.array(data_x).astype('float64')
        np.random.shuffle(reshaped_X)
        x = reshaped_X
    for i in range(len(y_trans) - sequence_length):
        temp_y = (y_trans[i: i + sequence_length])
        data_y.append(temp_y)
        reshaped_y = np.array(data_y).astype('float64')
        np.random.shuffle(reshaped_y)
        y = reshaped_y
    #数据支离
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
    #数据维度转化
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    #反归一化
    y_test = scaler.inverse_transform(y_test)
    return x_train, x_test, y_train, y_test

#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm(batch_size=20,time_step=10):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    x_train, x_test, y_train, y_test=deal_data('e:/RB_Ddata.csv',10)
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver()
    #module_file = tf.train.latest_checkpoint()    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #重复训练10000次
        for i in range(2000):
            for step in range(len(x_train)//batch_size-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:x_train[step*batch_size:(step+1)*batch_size],Y:y_train[step*batch_size:(step+1)*batch_size]})
                print(step,loss_)
            print(i,loss_)
            if i % 1==0:
                save_path = saver.save(sess, "d:/variables/basiclstm1.ckpt")
                print("model saved: %s, loss: %s" %(save_path, loss_))
                


train_lstm()


#————————————————预测模型————————————————————
def prediction(time_step=10):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    x_train, x_test, y_train, y_test=deal_data('e:/RB_Ddata.csv',10)
    y_test = y_test[:,-1]
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        saver = tf.train.Saver()
        os.path.exists('d:/variables/basiclstm1.ckpt')
        saver.restore(sess, 'd:/variables/basiclstm1.ckpt')
        
        test_predict=[]
        for step in range(len(x_test)):
            prob=sess.run(pred,feed_dict={X:[x_test[step]]})   
            test_predict.append(prob)
            
        test_predict = np.array(test_predict)
        scaler = MinMaxScaler()
        y_test = scaler.fit_transform(y_test)
        y_test = scaler.inverse_transform(y_test)
        test_predict = np.reshape(test_predict, [len(test_predict), time_step])[:,-1]
        test_predict = scaler.inverse_transform(test_predict)
        #test_y=np.array(test_y)*std[7]+mean[7]
        #test_predict=np.array(test_predict)*std[7]+mean[7]
        #acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(y_test))), y_test,  color='r')
        plt.show()

#prediction() 

#---------------------------------------------------------------------------------------------------------------------------------

def prediction():
    pred,_=lstm(1)    #预测时只输入[1,time_step,input_size]的测试数据
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint(base_path+'module2/')
        saver.restore(sess, module_file) 
        #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq=train_x[-1]
        predict=[]
        #得到之后100个预测结果
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()
