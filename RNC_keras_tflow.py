# Guilherme Afonso
# Código Baseado em Documentação da API Keras
# com algumas mudanças e adaptações

from matplotlib import pyplot
# Simples modelos de redes neurais convolucionais para a base de dados cifar 10
import time
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.externals._pilutil import toimage
# Importando multiprocessamento do tensorflow (a implementar)
import tensorflow as tf
import multiprocessing as mp

# apenas para dessativar algumas mensagens de log
import logging
logging.getLogger('tensorflow').disabled = True

# iniciando contagem de tempo
start = time.time()

K.set_image_dim_ordering('th')

# carregando o dataset cifar 10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# nome das classes presentes na base de dados
class_names = ['aviao', 'automovel', 'passaro', 'gato', 'cervo', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhao']
num_classes = 10 # número de classes

fig = plt.figure(figsize=(8, 3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:] == i)[0]
    features_idx = X_train[idx, ::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num, ::], (1, 2, 0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

# create a grid of 3x3 images
# for i in range(0, 9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(toimage(X_train[i]))
# # show the plot
# pyplot.show()

# uma semente aleatória para o random do numpy
seed = 7
np.random.seed(seed)

# normalização para a manipulação de imagem já que ela é recebida como INT
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# inicia y train e x test: recebe valor convertido de um vetor em matriz binária
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Criação do modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# executando modelo
# 1 - definindo numero de periodos de treinamento
# Fiz testes para 10, 20 ,30, 40 e 50
#epochs = 10
#epochs = 20
#epochs = 30
#epochs = 40
epochs = 50

lrate = 0.01
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model_info = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
rnc = model # rnc recebe modelo apenas para usar na

# avaliação final do modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# resultado da matriz de confusão
Y_pred = rnc.predict(X_test, verbose=2)
y_pred = np.argmax(Y_pred, axis=1)

end = time.time()
#print("tempo de execução %.0.2f%%" % (end - start))

for ix in range(10):
    print(ix, confusion_matrix(np.argmax(y_test, axis=1), y_pred)[ix].sum())
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(cm)


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # valores para acurácia
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Plot Acurácia')
    axs[0].set_ylabel('Acurácia')
    axs[0].set_xlabel('Períodos')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['Treino', 'Avaliação'], loc='best')
    # valores para perda
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Perda')
    axs[1].set_xlabel('Período')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['Treino', 'Avaliação'], loc='best')
    plt.show()


# Ver matriz de confusão
df_cm = pd.DataFrame(cm, range(10), range(10))
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
plt.show()

plot_model_history(model_info)
