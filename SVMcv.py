# -*- coding: utf-8 -*-
"""
SVMcv.py

Script de clasificación SVM. Parte de los datos de entrenamiento preproc.csv 
y de test preprocTest.csv a los que se les ha aplicado un preprocesamiento en
el script PreprocesamientoSVM.py.

Este script se diferencia de SVMKaggle.py en que realiza una validación 
cruzada estratificada con los datos de entrenamiento de Titanic Spaceship, 
para hacer una evaluacion más completa que con la precisión de Kaggle. 
Concretamente, se introducen precisión, especificidad, sensibilidad, 
precisión balanceada (media de especificidad y sensibilidad),
F1 Score, curva ROC y métrica AUC (en variable tmp_Results).


@author: José Manuel Nieto del Valle
"""

from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold                    
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

    """

    L_NumFeatures = np.size(X_train,axis=0)

    # Apply Mann-Whitney-Wilcoxon U-Test:
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

    # IDs no discasrdes son columnas ordenadas segun pvalus, no esta filtrada
    return X_train,X_test,IDs_NonDiscarded,PVal_NonDiscarded


def structConfMat(confmat, index=0, multiple=False):
    """
    Crea un dataframe de pandas a partir de una matriz de confusión.
    Distingue entre clasificación binaria o multiclase.
    
    Parameters
    ----------
    confmat : numpy.ndarray
        array de n filas, representa la matriz de confusión.
    index : INT, opcional
        índice del dataframe, por defecto es 0
    multiple : BOOL, optional
        si True, devuelve métricas por CV. Si falso, devuelve media y std de 
        las métricas.
        
    Returns
    -------
    performance : pd.DataFrame
        dataframe con las métricas de rendimiento.
        Example for latex tables:
            print(structConfMat(confmat,multiple=False)
            .to_latex(float_format="{0.real:.3} [{0.imag:.2}]".format))

        Note: for coonverting multiple performance to average/std use
            (performance.mean() + 1j*performance.std()).to_frame().T
    """

    intdim = int(np.sqrt(confmat.shape[1]))
    conf_n = confmat.reshape((len(confmat), intdim, intdim))
    corrects = conf_n.transpose(2,1,0).reshape((-1,len(conf_n)))[::(intdim+1)]
    corrects = corrects.sum(axis=0)
    n_folds = conf_n.sum(axis=1).sum(axis=1)
    cr = corrects/n_folds

    aux_n = conf_n[:,0][:,0]/conf_n[:,0].sum(axis=1)
    for ix in range(intdim-1):
        aux_n = np.c_[aux_n, conf_n[:,ix+1][:,ix+1]/conf_n[:,ix+1].sum(axis=1)]

    b_acc = np.nanmean(aux_n, axis=1)

    performance = pd.DataFrame({'CorrectRate': cr, 'ErrorRate': 1-cr,
                                'balAcc': b_acc},
                               index=index+np.arange(confmat.shape[0]))
    for ix in range(aux_n.shape[1]):
        auxperf = pd.DataFrame({f'Class_{ix}': aux_n[:,ix]},
                               index=index+np.arange(confmat.shape[0]))
        performance = pd.concat((performance, auxperf),axis=1)

    if intdim==2:
        columns = performance.columns.tolist()
        columns[columns.index('Class_0')]='Sensitivity'
        columns[columns.index('Class_1')]='Specificity'
        performance.columns = columns
        prec = aux_n[:,1]/(aux_n[:,1]+1-aux_n[:,0])
        f1 = 2*prec*aux_n[:,1]/(prec+aux_n[:,1])
        performance['Precision'] = prec
        performance['F1'] = f1


    if multiple==False:
        performance = (performance.mean(skipna=True)
                       + 1j*performance.std(skipna=True)).to_frame().T
    return performance


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
    
    param_grid={'C': [1, 10, 100],
                'gamma': [1, 0.02, 0.01, 'scale'],
                'kernel': ['rbf']}
    classifier=svm.SVC(class_weight='balanced', cache_size=800)
    gs=GridSearchCV(classifier, param_grid)
    clf = make_pipeline(
        preprocessing.StandardScaler(), 
        gs)
    
    # poner njobs
    
    
    
    clf.fit(X_train,Y_train)
    Y_predict = clf.predict(X_test)
    Scores = clf.decision_function(X_test)
    params = gs.best_params_

    return clf,Y_predict,Scores, params


#%% EJECUCION PROGRAMA PRINCIPAL %%#

# -- Semilla aleatoria -- #

np.random.seed(666)


BBDD = pd.read_csv('preproc.csv')
nombres = BBDD.columns

StackX = BBDD.iloc[:,0:len(BBDD.columns)-1]
StackY = BBDD.iloc[:,len(BBDD.columns)-1]

BBDD =  BBDD.astype(float)

StackX = np.asarray(StackX)
StackY = np.array(StackY)


# -- Generación de los Stacks -- #

IDs_NonDiscarded = np.arange(len(BBDD.columns)-1) 

# Cargar test
Xtest = pd.read_csv('preprocTest.csv')



# -- Definiendo el esquema de validación cruzada -- #

n_splits = 5
n_repeats = 2

# 10 CV -> Según Kohavi et al.
split_tt = RepeatedStratifiedKFold(n_splits = 10, 
                                   n_repeats = 5, 
                                   random_state=None) 

# -- Declaración de las variables de salida -- #

Predictions = np.zeros_like(StackY,dtype=float) # Predicciones 
True_Labels = np.zeros_like(StackY,dtype=float) # Respuestas
Sets_Scores = np.zeros_like(StackY,dtype=float) # Puntuaciones

# -- Parche #1 -- #

if 'confmat_aux' in locals(): # Elimino matriz de confusión, si existe
    del confmat_aux

# -- Variables que declaro para hacer un seguimiento de la simulación -- #

Contador_Iteracion = 0

# -- Bucle principal -- #

for train,test in split_tt.split(StackX,StackY):
    
    t = time.time()
  
    Contador_Iteracion = Contador_Iteracion + 1
    porcentaje = Contador_Iteracion/(n_splits * n_repeats) * 100
    print('Iteracion: '+str(Contador_Iteracion),
          '/',
          n_splits * n_repeats, 
          '(',porcentaje,'%)') # Para llevar la cuenta de l simulación

    # -- Generar los Stacks -- #
    Xtrain = StackX[train,:]
    Xtest = StackX[test,:]
    Ytrain = StackY[train]
    Ytest = StackY[test]

    # -- Aplicar la selección de características -- #
    XtrainFS, XtestFS, IDs_NonDiscardedFS, PVal_NonDiscardedFS = CF_FeatureSelection(Xtrain,Ytrain,Xtest,IDs_NonDiscarded)

    # -- Aplicar la clasificación -- #
    clf,Ypredict,Scores,params = CF_Classifier(XtrainFS,Ytrain,XtestFS)

    # -- Actualizar las predicciones -- #
    Predictions[test] = Ypredict
    True_Labels[test] = Ytest
    Sets_Scores[test] = Scores

    tmp_confmat_aux = metrics.confusion_matrix(Ytest,Ypredict,labels=np.unique(StackY)).flatten()

    if 'confmat_aux' not in locals():
        confmat_aux = tmp_confmat_aux
    else:
        confmat_aux = np.vstack((confmat_aux,tmp_confmat_aux))


    elapsed = time.time() - t   
    tiempo_restante = (n_splits * n_repeats)*elapsed - Contador_Iteracion*elapsed
    minutos_restante = tiempo_restante/60

    print('Quedan',minutos_restante,'minutos')
    print('### ------------------- ###')
    
    
# -- Generando los resultados -- #

Bucle_CV = structConfMat(confmat_aux,multiple=True)
# Los resultados finales se almacenarán en el DF tmp_Results
tmp_Results = structConfMat(confmat_aux,multiple=False)


# -- Generando los parámetros para las curvas ROC -- #

Yscores = 1/(1+np.exp(-Sets_Scores))
fpr, tpr, thresholds = metrics.roc_curve(True_Labels, Yscores)
auc = metrics.roc_auc_score(True_Labels, Yscores)

plt.figure()
plt.plot(fpr,tpr,'g', label='SVM')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

caracteristicas_ord = pd.DataFrame({'Caracteristicas': BBDD.columns[IDs_NonDiscardedFS], 'Pvalues': PVal_NonDiscardedFS})
caracteristicas_ord.to_latex('pval_caracteristicas.tex', index=False)
caracteristicas_ord.to_csv('pval_caracteristicas.csv', index=False)


