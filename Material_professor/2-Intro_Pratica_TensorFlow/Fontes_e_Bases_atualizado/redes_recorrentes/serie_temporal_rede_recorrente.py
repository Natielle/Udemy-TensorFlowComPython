import pandas as pd
base = pd.read_csv('petr4.csv')
base = base.dropna()

base = base.iloc[:,1].values
import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(base)

periodos = 30
previsao_futura = 1 # horizonte

X = base[0:(len(base) - (len(base) % periodos))]
X_batches = X.reshape(-1, periodos, 1)

y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]
y_batches = y.reshape(-1, periodos, 1)

X_teste = base[-(periodos + previsao_futura):]
X_teste = X_teste[:periodos]
X_teste = X_teste.reshape(-1, periodos, 1)
y_teste = base[-(periodos):]
y_teste = y_teste.reshape(-1, periodos, 1)

import tensorflow.compat.v1 as tf
tf.disable_eager_execution ()
tf.disable_v2_behavior()
tf.reset_default_graph()

entradas = 1
neuronios_oculta = 100
neuronios_saida = 1

xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

n_layers = 4
layers = [tf.keras.layers.LSTMCell(units=neuronios_oculta, activation=tf.nn.relu)
          for layer in range(n_layers)]

#keep_prob = tf.placeholder(tf.float32)  

multi_layer_cell = tf.keras.layers.StackedRNNCells(layers)

saida_rnn, _ = tf.nn.dynamic_rnn(multi_layer_cell, xph, dtype = tf.float32)

stacked_rnn_outputs = tf.reshape(saida_rnn, [-1, neuronios_oculta]) 
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, neuronios_saida)
outputs = tf.reshape(stacked_outputs, [-1, periodos, neuronios_saida])

erro = tf.reduce_mean(tf.square(outputs - yph))
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoca in range(1000):
        _, custo = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches})
        #_, custo = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches, keep_prob : 0.1})
        if epoca % 100 == 0:
            print(epoca + 1, ' erro: ', custo)
    
    previsoes = sess.run(outputs, feed_dict = {xph: X_teste})
    #previsoes = sess.run(outputs, feed_dict = {xph: X_teste, keep_prob : 0.1})
    
import numpy as np
y_teste.shape
y_teste2 = np.ravel(y_teste)

previsoes2 = np.ravel(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, previsoes2)

plt.plot(y_teste2, '*', markersize = 10, label = 'Valor real')
plt.plot(previsoes2, 'o', label = 'Previsões')
plt.legend()

plt.plot(y_teste2, label = 'Valor real')
plt.plot(y_teste2, 'w*', markersize = 10, color = 'red')
plt.plot(previsoes2, label = 'Previsões')
plt.legend()
























