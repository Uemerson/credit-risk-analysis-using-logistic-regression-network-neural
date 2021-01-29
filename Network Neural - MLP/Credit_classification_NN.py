import pandas as pd
import numpy as np


dataset = pd.read_csv('crx_manipulation_with_index.csv')

##############Substituindo valores não conhecidos por np.NaN##################

dataset_cleaned = dataset.replace('?',np.NaN)

###############Remover colunas com dados faltantes############################

dataset_cleaned.dropna(inplace=True)

######################Convertendo colunas para float##########################
dataset_cleaned['A1'] = pd.to_numeric(dataset_cleaned['A1'])
dataset_cleaned['A2'] = pd.to_numeric(dataset_cleaned['A2'])
dataset_cleaned['A14'] = pd.to_numeric(dataset_cleaned['A14'])

dataset_cleaned.drop('Unnamed: 0',axis=1,inplace=True)

####################Separação de X's e Y's e normalização#####################

X = dataset_cleaned.drop(['A16'],axis=1)
Y = pd.DataFrame(dataset_cleaned['A16'])

from sklearn.preprocessing import StandardScaler  
import joblib

StandardScaler = StandardScaler()
X_scaled = StandardScaler.fit_transform(X)
joblib.dump(StandardScaler,'StandardScaler.save')

######################Divisão de dados de treino e teste######################

testing_size = 0.3
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y,
                                                    test_size = testing_size)

#######################Criar arquitetura de rede neural#######################

def Classification_NN(X,Y,dropout):

    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout 
    from keras.callbacks import EarlyStopping
    import joblib
    from keras.layers.advanced_activations import LeakyReLU

    hidden_neurons_layer = int((X.shape[1]*2/3))

    size_input = X.shape[1]
    neurons_output = Y.shape[1]
        
#########################Divisão Treino_teste#################################    

    es = EarlyStopping(monitor='val_loss',min_delta=1e-3,verbose=1,patience=20)
    
    #def build_classifier(optimizer,neurons):
    classifier = Sequential()
    ######################Primeira camada oculta###############################
    classifier.add(Dense(units = hidden_neurons_layer,
                         kernel_initializer = 'uniform',
                         activation = 'relu',
                         input_dim = size_input))

    classifier.add(Dropout(p= dropout))

    #####################Segunda camada oculta#################################
    
    classifier.add(Dense(units = hidden_neurons_layer,
                         kernel_initializer = 'uniform',
                         activation = 'relu'))
    classifier.add(Dropout(p= dropout))
  
    ###########################Camada de saída#################################
    classifier.add(Dense(units = neurons_output, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['accuracy'])
    return classifier,es

classifier,es = Classification_NN(X_test,Y_test,0.3)

classifier.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=32,epochs=10000,
                        use_multiprocessing = True,verbose=2,callbacks = [es]) 
Y_pred = classifier.predict(X_test)

#######################Binarizar todos os resultados em 0e 1##################

for i in range(len(Y_pred)):
    if Y_pred[i] >= 0.5:
        Y_pred[i] = 1
    else:
        Y_pred[i] = 0

###########################Análise das matrizes de confusão####################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

Result = pd.DataFrame(confusion_matrix(Y_test,Y_pred))
Result.to_csv('Output_confusion_matrix.csv')
Classification = classification_report(Y_test,Y_pred)


