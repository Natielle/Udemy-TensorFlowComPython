# -*- coding: utf-8 -*-
"""
Editor Spyder
Autor: Natielle Gonçalves de Sá
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

# Lendo os dados do banco
base = pd.read_csv('petr4.csv')
print(base.shape)

# Retirando as linhas que possuem NA
base = base.dropna()
print(base.shape)

# Observando o comportamento da ação
base = base.iloc[:,1].values # obtem os dados da colunas open (quanto abriu a bolsa)

#%matplotlib inline
plt.plot(base)

# Agora vamos criar variáveis para definirmos as nossas previsões
periodos = 30 # com base em 30 dias anteriores, vamos prever os 30 dias seguintes
previsao_futura = 1 # 1 vez os 30 dias

# ---------------------------------------------
#     SEPARANDO A BASE DE DADOS DE TREINO 
# ---------------------------------------------
# Qual parte de dados iremos utilizar
print("Linhas da base: ", len(base))
print("(len(base) % periodos): ", (len(base) % periodos))
print("len(base) - (len(base) % periodos): ", (len(base) - (len(base) % periodos)))
X = base[0:(len(base) - (len(base) % periodos))]
print("Tipo de X: ", type(X)) # iremos utilizar X para o treinamento
X_batches = X.reshape(-1, periodos, 1) # Qtde de registros, quanto tempo queremos prever, qtde de atributos
print("X_batches: ", X_batches)


# Separando os labels
print("Indice onde o y vai comecar: ", (len(base) - (len(base) % periodos)) + previsao_futura)
Y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]
Y_batches = Y.reshape(-1, periodos, 1)


# ---------------------------------------------
#     SEPARANDO A BASE DE DADOS DE TESTE 
# ---------------------------------------------
X_teste = base[-(periodos + previsao_futura):]
X_teste = X_teste[:periodos]
X_teste = X_teste.reshape(-1, periodos, 1)
Y_teste = base[-(periodos):]
Y_teste = Y_teste.reshape(-1, periodos, 1)


# ---------------------------------------------
#   APLICANDO AS REDES NEURAIS RECORRENTES 
# ---------------------------------------------
tf.reset_default_graph() # Bom limpar a memória

# Definindo as quantidade de neurônios
entradas = 1             # Teremos apenas 1 entrada
neuronios_oculta = 100   # Quantidade de neurônios na camada oculta
neuronios_saida = 1      # Queremos apenas uma resposta (que é quanto estará a ação)

# Criando os placeholders para receber os dados depois
xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

# Criando uma rede neural recorrente básica (que tem como função apenas a tangente hiperbólica)
celula = tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, 
                                     activation = tf.nn.relu)
# Mapeando os neurônios da camada oculta para ter apenas um neurônio de saída
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, 
                                                output_size = 1)


# Criando formula para obter as saídas
saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)

# Obtendo o erro 
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)

# Vamos usar o Adam Optimizer para otimizar/minimizar o erro
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)
print("Finalizamos de fazer a estrutura da rede.")

# ---------------------------------------------
#    TREINANDO A REDE NEURAL RECORRENTE 
# ---------------------------------------------
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Realiza o treinamento (minimizando o erro)
    for epoca in range(1000):
        _, custo = sess.run([treinamento, erro], 
                            feed_dict = {xph: X_batches, 
                                         yph: Y_batches})
        if epoca % 100 == 0:
            print(epoca + 1, ' erro: ', custo)
    
    # Caso queiramos realizar a previsão
    # Notar que a previsão só é feita depois do treinamento
    previsoes = sess.run(saida_rnn, 
                         feed_dict = {xph: X_teste})
    

# ---------------------------------------------
#          VISUALIZANDO OS RESULTADOS 
# ---------------------------------------------
# Iremos precisar mudar a dimensão de Y_teste para que fique mais fácil a comparação com as previsões
print("Dimensões do Y_teste: ", Y_teste.shape)
Y_teste2 = np.ravel(Y_teste) # Retorna um vetor flattened
print("Dimensões do Y_teste após ravel: ", Y_teste2.shape)
previsoes2 = np.ravel(previsoes)

# Calculando o erro absoluto (ou seja, retorna quanto de dinheiro o algoritmo está errando +-)
mae = mean_absolute_error(Y_teste2, previsoes2)
print("Erro absoluto: ", mae)

# analisando a diferença entre os valores reais e as previsões (gráfico de pontos)
# plt.clf() # limpa o gráfico antigo
# plt.plot(Y_teste2, '*', markersize = 10, label = 'Valor real')
# plt.plot(previsoes2, 'o', label = 'Previsões')
# plt.legend()

# analisando a diferença entre os valores reais e as previsões (gráfico de linhas)
plt.clf() # limpa o gráfico antigo
plt.plot(Y_teste2, label = 'Valor real')
# plt.plot(Y_teste2, 'w*', markersize = 10, color = 'red') # adiciona estrela nos pontos exatos
plt.plot(previsoes2, label = 'Previsões')
plt.legend()












