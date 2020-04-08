# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:18:27 2019

@author: Federico Checozzi
"""

import glob
#import os
import pandas as pd
import scipy as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#PROCESAMIENTO DE DATOS
#carga de datos, es mucha magia pero funciona
files = glob.glob('*/*/*.csv')
datalabels = {f : f.split("\\") for f in files}

# spectredata = pd.concat([pd.read_csv(file, sep = ';', header = 1, decimal = ',', usecols = [0, 1])\
#                              .assign(Group = labels[0],Sample = labels[1] ,File = labels[2])\
#                         for file,labels in datalabels.items()], ignore_index=True)\
#                 .pipe(pd.pivot_table, values="Counts", index=["Group","Sample","File"],columns=["Wavelength [nm]"])
                
spectredata = pd.concat(
                            [pd.read_csv(file, sep = ';', header = 1, decimal = ',', usecols = [0, 1], names = ["Wavelength","Intensity"])
                             .assign(Group = labels[0],Sample = labels[1] ,Measurement = labels[2].rstrip(".csv"))
                             for file,labels in datalabels.items()], ignore_index=True
                        ).pivot_table(
                            values="Intensity", index=["Group","Sample","Measurement"],columns=["Wavelength"]
                        )
    
#acceso a datos se realiza escribiendo "spectredata.at[('12_02', 'M5', 'M5_20.csv'),293.83]" por ejemplo

#normalizamos los datos
scaler=StandardScaler()
scaler.fit(spectredata)
spectres_scaled=scaler.transform(spectredata)

#generación de etiquetas de clase numéricas para cargar en el LDA
classstring = spectredata.reset_index(level = 0).Group
le = LabelEncoder()
classcode = le.fit_transform(classstring)

#separación entre entrenamiento y muestras de prueba
#X_train, X_test, y_train, y_test = train_test_split(spectres_scaled, classcode, test_size=0.2, random_state=0) 
    
#LDA
lda=LDA(n_components=3) 
#lda.fit(X_train,y_train)
#Xlda_train=lda.transform(X_train)
lda.fit(spectres_scaled,classcode)
spectres_lda=lda.transform(spectres_scaled)
score = 100*lda.score(spectres_scaled,classcode)

#Por desgracia scikit devuelve arrays, así que creo un dataframe con los resultados del PCA y los índices de los datos originales
#df_lda =  pd.DataFrame(data = Xlda_train, columns=["LDA" + str(i) for i in range(1,4)])
df_lda =  pd.DataFrame(data = spectres_lda, columns=["LDA" + str(i) for i in range(1,4)])
df_lda.index = spectredata.index

#GRÁFICOS
ldapalette = ['blue'  , 'orange' , 'red' , 'green']
ldamarkers = ['o' , '^' , 's' , 'v']

df2 = df_lda.reset_index(level = 0) #esta línea convierte uno de los índices en una columna porque mucho código funciona más fácil así
ax = sns.scatterplot(data = df2, x = "LDA1", y = "LDA2", hue = 'Group', palette = ldapalette, style = 'Group', markers = ldamarkers)
ax.set_title("LDA archivos .spc, precisión = %2.2f %%"%score)
ax.set_xlabel("LDA1")
ax.set_ylabel("LDA2")

nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = sp.meshgrid(sp.linspace(x_min, x_max, nx),
                     sp.linspace(y_min, y_max, ny))
#Z = lda.predict_proba(sp.c_[xx.ravel(), yy.ravel()])#conceptualmente equivocado
#Z = Z[:, 1].reshape(xx.shape)