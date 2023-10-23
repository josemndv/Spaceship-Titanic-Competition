# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:55:04 2022

Preprocesamiento de Titanic Spaceship

@author: outat
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

########### TRAIN #############################################################

titanic = pd.read_csv("train.csv") # Conjunto de entrenamiento


# --- Imputación --- #

# Las variables Age, RoomService, FoodCourt, ShoppingMall, Spa, y VRDeck se 
# imputarán mediante el algoritmo KNN
imputerKNN = KNNImputer(n_neighbors=5, weights="uniform")

# Age
imputerKNN.fit(titanic[["Age"]])
titanic["Age"] = imputerKNN.transform(
    titanic[["Age"]]).ravel()

# RoomService
imputerKNN.fit(titanic[["RoomService"]])
titanic["RoomService"] = imputerKNN.transform(
    titanic[["RoomService"]]).ravel()

# FoodCourt
imputerKNN.fit(titanic[["FoodCourt"]])
titanic["FoodCourt"] = imputerKNN.transform(
    titanic[["FoodCourt"]]).ravel()

# ShoppingMall
imputerKNN.fit(titanic[["ShoppingMall"]])
titanic["ShoppingMall"] = imputerKNN.transform(
    titanic[["ShoppingMall"]]).ravel()

# Spa
imputerKNN.fit(titanic[["Spa"]])
titanic["Spa"] = imputerKNN.transform(
    titanic[["Spa"]]).ravel()

# VRDeck
imputerKNN.fit(titanic[["VRDeck"]])
titanic["VRDeck"] = imputerKNN.transform(
    titanic[["VRDeck"]]).ravel()


# Las variables categóricas hay que imputarlas mediante otro algoritmo. Se 
# propone hacer la sustitución con el más frecuente o con la media de cada 
# variable

# [!] Hacer método que calcule automáticamente, no a ojo
# cabinDefault = str(titanic['Cabin'].mode())
cabinDefault = 'G/734/S'
# homeDefault = str(titanic['HomePlanet'].mode())
homeDefault = 'Earth'
# cryoDefault = str(titanic['CryoSleep'].mode())
cryoDefault = False
# destinationDefault = str(titanic['Destination'].mode())
destinationDefault = 'TRAPPIST-1e'
# vipDefault = str(titanic['VIP'].mode())
vipDefault = False

# Imputación
titanic.loc[titanic['Cabin'].isna(), 'Cabin'] = cabinDefault
titanic.loc[titanic['HomePlanet'].isna(), 'HomePlanet'] = homeDefault
titanic.loc[titanic['CryoSleep'].isna(), 'CryoSleep'] = cryoDefault
titanic.loc[titanic['Destination'].isna(), 'Destination'] = destinationDefault
titanic.loc[titanic['VIP'].isna(), 'VIP'] = vipDefault


# Bucle que separa los strings de la variable Cabin en tres nuevas variables.
# P. ej. G/734/S se dividiría en Zone: G, Room: 734, Zone2: S
for i in range(len(titanic.index)):
    index = titanic.index[i] # Con nueva modf se puede usar i directamente
    cabin = titanic.loc[index, 'Cabin'] # String
    print(index)
    print(cabin.split("/", 2))
    if isinstance(cabin, float) == False:
        a = cabin.split("/", 2)
        titanic.loc[index, 'Zone'] = a[0]
        titanic.loc[index, 'Room'] = a[1]
        titanic.loc[index, 'Zone2'] = a[2]
    


group_size = np.zeros(len(titanic.index))
# for pid in range(len(titanic.index)):
    # var = titanic.loc[pid, 'PassengerId']
res = []
running_count = 1
for i in range(len(titanic.index)-1):
    if titanic.loc[i, 'PassengerId'].split("_")[0] == titanic.loc[i+1, 'PassengerId'].split("_")[0]:
        running_count += 1
    else:
        res.extend([running_count]*running_count)
        running_count = 1
# res.append(res[len(titanic.index)-2])
# group_size[pid] = running_count

# ChAPUZA DEL SIGLO
res.append(2)
res.append(2)

# titanic['GroupSize'] = GroupSize(titanic['PassengerId'])

# titanic['GroupSize'] = titanic.assign(GroupSize = GroupSize(titanic.PassengerId))

titanic['GroupSize'] = res

length = np.zeros(len(titanic.index))
for i in range(len(titanic.index)-1):
    length[i] = len(str(titanic.loc[i, 'Name']))

titanic['Length'] = length


# Hacemos dummys de algunas variables
titanicDUMMY= pd.get_dummies(titanic, 
                             columns = ['HomePlanet', 'Destination', 
                                        'CryoSleep', 'VIP', 'Zone', 
                                        'Zone2', 'Transported'])



# Desechamos dummys redundantes
titanicSelect = titanicDUMMY.drop(columns = ['PassengerId', 'Cabin', 'Name', 
                                             'CryoSleep_False', 'VIP_False', 
                                             'Zone2_S', 'Transported_False'])




# Se guarda el DF resultante en preproc.csv sin guardar los índices
titanicSelect.to_csv('preproc.csv', index=False)




########### TEST ##############################################################

titanicTest = pd.read_csv("test.csv") # Conjunto test


# --- Imputación --- #

# Las variables Age, RoomService, FoodCourt, ShoppingMall, Spa, y VRDeck se 
# imputarán mediante el algoritmo KNN
imputerKNN = KNNImputer(n_neighbors=5, weights="uniform")

# Ajustamos el modelo e imputamos los missing values
imputerKNN.fit(titanicTest[["Age"]])
titanicTest["Age"] = imputerKNN.transform(
    titanicTest[["Age"]]).ravel()

imputerKNN.fit(titanicTest[["RoomService"]])
titanicTest["RoomService"] = imputerKNN.transform(
    titanicTest[["RoomService"]]).ravel()

imputerKNN.fit(titanicTest[["FoodCourt"]])
titanicTest["FoodCourt"] = imputerKNN.transform(
    titanicTest[["FoodCourt"]]).ravel()

imputerKNN.fit(titanicTest[["ShoppingMall"]])
titanicTest["ShoppingMall"] = imputerKNN.transform(
    titanicTest[["ShoppingMall"]]).ravel()

imputerKNN.fit(titanicTest[["Spa"]])
titanicTest["Spa"] = imputerKNN.transform(
    titanicTest[["Spa"]]).ravel()

imputerKNN.fit(titanicTest[["VRDeck"]])
titanicTest["VRDeck"] = imputerKNN.transform(
    titanicTest[["VRDeck"]]).ravel()

# Las variables categóricas hay que imputarlas mediante otro algoritmo. Se 
# propone hacer la sustitución con el más frecuente o con la media de cada 
# variable

# [!] Hacer método que calcule automáticamente, no a ojo
cabinDefault = 'G/734/S'
homeDefault = 'Earth'
cryoDefault = False
destinationDefault = 'TRAPPIST-1e'
vipDefault = False


titanicTest.loc[titanicTest['Cabin'].isna(), 
                'Cabin'] = cabinDefault
titanicTest.loc[titanicTest['HomePlanet'].isna(), 
                'HomePlanet'] = homeDefault
titanicTest.loc[titanicTest['CryoSleep'].isna(), 
                'CryoSleep'] = cryoDefault
titanicTest.loc[titanicTest['Destination'].isna(), 
                'Destination'] = destinationDefault
titanicTest.loc[titanicTest['VIP'].isna(), 
                'VIP'] = vipDefault


# Bucle que separa los strings de la variable Cabin en tres nuevas variables.
# P. ej. G/734/S se dividiría en Zone: G, Room: 734, Zone2: S
for i in range(len(titanicTest.index)):
    index = titanicTest.index[i] # Con nueva modf se puede usar i directamente
    cabin = titanicTest.loc[index, 'Cabin'] # String
    print(index)
    print(cabin.split("/", 2))
    if isinstance(cabin, float) == False:
        a = cabin.split("/", 2)
        titanicTest.loc[index, 'Zone'] = a[0]
        titanicTest.loc[index, 'Room'] = a[1]
        titanicTest.loc[index, 'Zone2'] = a[2]
    
 
group_size = np.zeros(len(titanicTest.index))
# for pid in range(len(titanic.index)):
    # var = titanic.loc[pid, 'PassengerId']
res = []
running_count = 1
for i in range(len(titanicTest.index)-1):
    if titanicTest.loc[i, 'PassengerId'].split("_")[0] == titanicTest.loc[i+1, 'PassengerId'].split("_")[0]:
        running_count += 1
    else:
        res.extend([running_count]*running_count)
        running_count = 1
# res.append(res[len(titanic.index)-2])
# group_size[pid] = running_count

# ChAPUZA DEL SIGLO
res.append(1)

# titanic['GroupSize'] = GroupSize(titanic['PassengerId'])

# titanic['GroupSize'] = titanic.assign(GroupSize = GroupSize(titanic.PassengerId))

titanicTest['GroupSize'] = res   
 
length = np.zeros(len(titanicTest.index))
for i in range(len(titanicTest.index)-1):
    length[i] = len(str(titanicTest.loc[i, 'Name']))

titanicTest['Length'] = length

 
# Hacemos dummys de algunas variables
titanicDUMMYTest = pd.get_dummies(titanicTest,  
                                  columns = ['HomePlanet', 'Destination', 
                                             'CryoSleep', 'VIP', 'Zone', 
                                             'Zone2'])





# Desechamos dummys redundantes
titanicSelectTest = titanicDUMMYTest.drop(columns = ['PassengerId', 'Cabin', 
                                                     'Name', 'CryoSleep_False',
                                                     'VIP_False', 'Zone2_S'])


# Se guarda el DF de test resultante en preprocTest.csv sin guardar los índices
titanicSelectTest.to_csv('preprocTest.csv', index=False)





