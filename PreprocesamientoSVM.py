# -*- coding: utf-8 -*-
"""
Preprocesamiento SVM

Script de preprocesamiento del dataset Titanic Spaceship. Devuelve un fichero
.csv con las variables preprocesadas para pasar a realizar una clasificación
en los archivos SVMKaggle.py (evaluando con los datos de test), o SVMcv.py (con
los datos de entrenamiento con validación cruzada estratificada).

@author: José Manuel Nieto del Valle
"""
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import seaborn as sns


# %% TRAIN/TEST


test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

print(train.isnull().sum())
sns.heatmap(train.isnull())

print(test.isnull().sum())
sns.heatmap(test.isnull())

train_test=train.append(test, ignore_index= True)

print(train_test.isnull().sum())
sns.heatmap(train_test.isnull())


# --- Imputaciones lógicas --- #

# En primer lugar deben ir las imputaciones lógicas: si duerme no gasta, 
# si gasta no duerme, etc
# Si duermes no gastas dinero. Completemos el costo faltante con "0" si un 
# pasajero estaba en CryoSleep.  

expenses_columns = ['RoomService', 
                    'FoodCourt', 
                    'ShoppingMall', 
                    'Spa', 
                    'VRDeck']

expensesRSpaVR_columns = ['RoomService',
                          'Spa', 
                          'VRDeck']


train_test.loc[:, expenses_columns] = train_test.apply(
    lambda x: 0 if x.CryoSleep == True else x, axis = 1)


# De igual forma, suponemos que si no se ha gastado dinero el pasajero        
# se encuentra con mucha  probabilidad en el estado de CryoSleep                                             
# Creamos una columna adicional que sume los gastos totales, "Expenses"   
# Expenses =  RoomService + FoodCourt + ShoppingMall + Spa + VRDeck    
# Y una que sume los gastos de habitación , spa y VR
# ExpensesRSpaVR =  RoomService + Spa + VRDeck  

train_test['Expenses'] = train_test.loc[:,expenses_columns].sum(axis=1)
train_test['ExpensesRSpaVR'] = train_test.loc[:,expensesRSpaVR_columns].sum(axis=1)
# No spending
train_test['No_spending']=(train_test['Expenses']==0).astype(int)

train_test.loc[:, ['CryoSleep']] = train_test.apply(
    lambda x: True if x.Expenses == 0 and pd.isna(x.CryoSleep) else x, axis = 1)
train_test.loc[:, ['CryoSleep']] = train_test.apply(
    lambda x: False if x.Expenses > 0 and pd.isna(x.CryoSleep) else x, axis = 1)


# Las variables demás variables Age, RoomService, FoodCourt, ShoppingMall,
# Spa, y VRDeck imputarán mediante el algoritmo KNN. Cabin, homePlanet, 
# cryoSleep, destination, y VIP se imputarán con el más frecuente (moda)



# --- Imputaciones KNN--- #

# Las variables Age, RoomService, FoodCourt, ShoppingMall, Spa, y VRDeck se 
# imputarán mediante el algoritmo KNN
imputerKNN = KNNImputer(n_neighbors=5, weights="uniform")

# Age
imputerKNN.fit(train_test[["Age"]])
train_test["Age"] = imputerKNN.transform(
    train_test[["Age"]]).ravel()

# RoomService
imputerKNN.fit(train_test[["RoomService"]])
train_test["RoomService"] = imputerKNN.transform(
    train_test[["RoomService"]]).ravel()

# FoodCourt
imputerKNN.fit(train_test[["FoodCourt"]])
train_test["FoodCourt"] = imputerKNN.transform(
    train_test[["FoodCourt"]]).ravel()

# ShoppingMall
imputerKNN.fit(train_test[["ShoppingMall"]])
train_test["ShoppingMall"] = imputerKNN.transform(
    train_test[["ShoppingMall"]]).ravel()

# Spa
imputerKNN.fit(train_test[["Spa"]])
train_test["Spa"] = imputerKNN.transform(
    train_test[["Spa"]]).ravel()

# VRDeck
imputerKNN.fit(train_test[["VRDeck"]])
train_test["VRDeck"] = imputerKNN.transform(
    train_test[["VRDeck"]]).ravel()


# Valores por defecto: moda
cabinDefault = 'G/734/S'
homeDefault = 'Earth'
cryoDefault = False
destinationDefault = 'TRAPPIST-1e'
vipDefault = False


# Imputación
train_test.loc[train_test['Cabin'].isna(), 
               'Cabin'] = cabinDefault
train_test.loc[train_test['HomePlanet'].isna(), 
               'HomePlanet'] = homeDefault
train_test.loc[train_test['CryoSleep'].isna(), 
               'CryoSleep'] = cryoDefault
train_test.loc[train_test['Destination'].isna(), 
               'Destination'] = destinationDefault
train_test.loc[train_test['VIP'].isna(), 
               'VIP'] = vipDefault


# Bucle que separa los strings de la variable Cabin en tres nuevas variables.
# P. ej. G/734/S se dividiría en Zone: G, Room: 734, Zone2: S
for i in range(len(train_test.index)):
    index = train_test.index[i] 
    cabin = train_test.loc[index, 'Cabin'] 
    if isinstance(cabin, float) == False:
        a = cabin.split("/", 2)
        train_test.loc[index, 'Deck'] = a[0]
        train_test.loc[index, 'Room'] = a[1]
        train_test.loc[index, 'Side'] = a[2]
    

    
# Los decks A, B, C o T vienen del planeta Europa
train_test.loc[(train_test['HomePlanet'].isna()) & (train_test['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet']='Europa'

# Los pasajeros del Deck G vienen de Earth
train_test.loc[(train_test['HomePlanet'].isna()) & (train_test['Deck']=='G'), 'HomePlanet']='Earth'
 
# Separación en nombre y apellido
train_test['Surname']=train_test['Name'].str.split().str[-1]

SHP_gb=train_test.groupby(['Surname','HomePlanet'])['HomePlanet'].size().unstack().fillna(0)

# Cuando la variable HomePlanet no existe se imputa con el nombre de su familia
SHP_index=train_test[train_test['HomePlanet'].isna()][(train_test[train_test['HomePlanet'].isna()]['Surname']).isin(SHP_gb.index)].index

train_test.loc[SHP_index,'HomePlanet']=train_test.iloc[SHP_index,:]['Surname'].map(lambda x: SHP_gb.idxmax(axis=1)[x])



# Ocupación en la cabina de cada pasajero   
group_size = np.zeros(len(train_test.index))
res = []
gente = 1
for i in range(len(train_test.index)-1):
    if train_test.loc[i, 'PassengerId'].split("_")[0] == train_test.loc[i+1, 'PassengerId'].split("_")[0]:
        gente += 1
    else:
        res.extend([gente]*gente)
        gente = 1

res.append(1)
train_test['GroupSize'] = res   
    


# -- Separar train y test --- #

train = train_test[train_test['Transported'].notnull()].copy()
test = train_test[train_test['Transported'].isnull()].drop("Transported",
                                                           axis=1)
test = test.reset_index(drop=True) # Reseteo el índice desde 0

# Hacemos dummys de algunas variables
trainDummy = pd.get_dummies(train, 
                              columns = ['HomePlanet', 'Destination', 
                                        'CryoSleep', 'VIP', 'Deck', 
                                        'Side', 'Transported'])

testDummy = pd.get_dummies(test, 
                              columns = ['HomePlanet', 'Destination', 
                                        'CryoSleep', 'VIP', 'Deck', 
                                        'Side'])



# Desechamos dummys redundantes
trainSelect = trainDummy.drop(columns = ['PassengerId', 'Cabin', 'Name', 
                                              'CryoSleep_False', 'VIP_False', 
                                              'Side_S', 'Transported_False',
                                              'Surname'])

testSelect = testDummy.drop(columns = ['PassengerId', 'Cabin', 'Name', 
                                              'CryoSleep_False', 'VIP_False', 
                                              'Side_S', 'Surname'])



# Se guarda el DF resultante en preproc.csv sin guardar los índices
trainSelect.to_csv('preproc.csv', index=False)
testSelect.to_csv('preprocTest.csv', index=False)








