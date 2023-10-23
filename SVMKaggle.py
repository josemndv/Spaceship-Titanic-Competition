# -*- coding: utf-8 -*-
"""
SVMKaggle.py

Script de clasificación SVM. Parte de los datos de entrenamiento preproc.csv 
y de test preprocTest.csv a los que se les ha aplicado un preprocesamiento en
el script PreprocesamientoSVM.py.

Devuelve las predicciones en el formato especificado por Kaggle con los envíos,
con un algoritmo SVM ajustado con los mejores hiperparámetros.


@author: José Manuel Nieto del Valle
"""

from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn import preprocessing, svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

#%% FUNCIONES %%#


def CF_FeatureSelection(X_train,Y_train,X_test,IDs_NonDiscarded):
    """
    Función para seleccionar las caracteristicas más relevantes con el test de 
    Mann-Whitney-Wilcoxon.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.
    IDs_NonDiscarded : Array of int32
        índices de las variables.

    Returns
    -------
    X_train : float64
        datos de entrenamiento tras selección.
    X_test : float64
        datos de test tras selección.
    IDs_NonDiscarded : Array of int32
        índices de las variables.
    PVal_NonDiscarded : Array of float64
        pvalores de las variables.

    ""
    Función para seleccionar las caracteristicas más relevantes con el test de 
    Mann-Whitney-Wilcoxon.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiqetas de entrenamiento.
    X_test : Array of float64
        datos de test.
    IDs_NonDiscarded : Array of int32
        índices de las variables.

    Returns
    -------
    X_train : float64
        datos de entrenamiento tras selección.
    X_test : float64
        datos de test tras selección.
    IDs_NonDiscarded : Array of int32
        índices de las variables.
    PVal_NonDiscarded : Array of float64
        pvalores de las variables.

    """
    
    L_NumFeatures = np.size(X_train,axis=0)

    # Mann-Whitney-Wilcoxon U-Test:
    pvalues=[]
    statistics=[]
    for col in np.arange(np.size(X_train,axis=1)):
        dataX=X_train[:,col]
        dataY=Y_train
        res_statistic, res_pvalue = stats.mannwhitneyu(dataX,dataY)
        pvalues.append(res_pvalue)
        statistics.append(res_statistic)
    pvalues = np.array(pvalues)
    orden = np.argsort(pvalues)

    orden = orden[0:L_NumFeatures-1]
    IDs_NonDiscarded = IDs_NonDiscarded[orden]
    PVal_NonDiscarded = pvalues[orden]

    X_train = X_train[:,orden]
    X_test = X_test[:,orden]

    selectedpvalues = np.concatenate(np.where(pvalues < 0.05))
    X_train = X_train[:, selectedpvalues]
    X_test = X_test[:, selectedpvalues]

    # IDs_NonDiscarded son los índices de las columnas ordenadas segun pvalues
    # IDs_NonDiscarded no esta filtrado pero Xtrain si
    return X_train,X_test,IDs_NonDiscarded,PVal_NonDiscarded


def CF_Classifier(X_train,Y_train,X_test):
    """
    Bloque de clasificación.

    Parameters
    ----------
    X_train : Array of float64
        datos de entrenamiento.
    Y_train : Array of int64
        etiquetas de entrenamiento.
    X_test : Array of float64
        datos de test.

    Returns
    -------
    clf : pipeline
        pipeline de clasificación.
    Y_predict : Array of int64
        predicciones sobre los datos de test.
    Scores : Array of float64
        puntuacines del clasificador.
    params : dict
        mejores parámetros que se han obtenido.


    """
    
    
    param_grid={'C': [1, 10, 100, 1000],
                'gamma': [1, 0.02, 0.01, 0.001, 'scale'],
                'kernel': ['rbf']}
    classifier=svm.SVC(class_weight='balanced', cache_size=900)
    gs=GridSearchCV(classifier, param_grid)
    clf = make_pipeline(
        preprocessing.StandardScaler(), 
        gs)


    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test) # Predicciones
    Scores = clf.decision_function(X_test)
    params = gs.best_params_ 

    return clf, Y_predict, Scores, params




#%% EJECUCION PROGRAMA PRINCIPAL %%#

# -- Semilla aleatoria -- #

np.random.seed(666)

# -- Cargar train y test -- #

BBDD = pd.read_csv('preproc.csv') # Datos preprocesados de train
Xtest = pd.read_csv('preprocTest.csv') # Datos preprocesados de test

StackX = BBDD.iloc[:,0:len(BBDD.columns)-1]
StackY = BBDD.iloc[:,len(BBDD.columns)-1]

Xtrain = np.asarray(StackX)
Ytrain = np.array(StackY)


IDs_NonDiscarded = np.arange(len(BBDD.columns)-1) # Indice de cada columna


# Extrae la columna PassengerId por separado para crear DF de predicciones
titanicIds = pd.read_csv("test.csv")
resultados = titanicIds[["PassengerId"]]
resultados = pd.DataFrame(resultados)

# -- Aplicar la selección de características -- #
Xtest = np.asarray(Xtest)
XtrainFS, XtestFS, IDs_NonDiscardedFS, PVal_NonDiscardedFS = CF_FeatureSelection(Xtrain, 
                                                                                 Ytrain, 
                                                                                 Xtest, 
                                                                                 IDs_NonDiscarded)

# -- Aplicar la clasificación -- #

clf, Ypredict, Scores, params = CF_Classifier(XtrainFS,StackY,XtestFS)
# params contiene los mejores hiperparámetros de SVM
# MEJOR 0.80734 (Kaggle) -> gamma=0.02, C=10, kernel = rbf
    
predsFinal = np.array(Ypredict, dtype=bool)


# -- Guardar resultados -- #

resultados['Transported'] = predsFinal.tolist()

now = datetime.now()
# El nombre de cada resultado tendrá la fecha de su generación
resultados.to_csv('Resultados/' + now.strftime("%d_%m_%Y-%H_%M_%S") + '.csv', index=False)

# Creamos un DF con las características y sus pvalues
caracteristicas_ord = pd.DataFrame({'Caracteristicas': BBDD.columns[IDs_NonDiscardedFS], 'Pvalues': PVal_NonDiscardedFS})
caracteristicas_ord.to_latex(index=False)
