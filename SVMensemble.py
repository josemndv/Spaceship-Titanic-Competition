# -*- coding: utf-8 -*-
"""
Script que implementa la técnica Weighted Exponential Ensemble, teniendo en 
cuenta un número de resultados de clasificación, para mejorar los resultados
finales.

@author: Usuario
"""

import numpy as np 
import pandas as pd

sub = pd.read_csv('Resultados/15_01_2023-13_37_58.csv')
sub.sort_values(by=['PassengerId'], inplace=True)
sub['Transported'] = sub['Transported'].astype('float')


sub1 = pd.read_csv('Resultados/15_01_2023-13_29_55.csv')
sub1.sort_values(by=['PassengerId'], inplace=True)
sub1['Transported'] = sub1['Transported'].astype('float')
sub2 = pd.read_csv('Resultados/12_01_2023-19_17_05.csv')
sub2.sort_values(by=['PassengerId'], inplace=True)
sub2['Transported'] = sub2['Transported'].astype('float')
sub3 = pd.read_csv('Resultados/11_01_2023-16_30_54.csv')
sub3.sort_values(by=['PassengerId'], inplace=True)
sub3['Transported'] = sub3['Transported'].astype('float')

b = 300.0
S = 0.80734
q = 0.0

sub['Transported'] = sub1['Transported']*np.exp(b*(0.80734-S))
q = q + np.exp(b*(0.80734-S))
sub['Transported'] = sub['Transported'] + sub2['Transported']*np.exp(b*(0.805-S))
q = q + np.exp(b*(0.805-S))
sub['Transported'] = sub['Transported'] + sub3['Transported']*np.exp(b*(0.79658-S))
q = q + np.exp(b*(0.79658-S))

print(q)
sub.head(10)

sub['Transported'] = np.rint(sub['Transported'])
sub.head(10)

sub['Transported'] = sub['Transported'].astype('bool')
sub.to_csv('resultEWE.csv', index=False)
sub.head(10)