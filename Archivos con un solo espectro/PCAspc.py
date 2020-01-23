# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:18:27 2019

@author: Federico Checozzi
"""

import glob
#import os
import pandas as pd
#import scipy as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#PROCESAMIENTO DE DATOS
#carga de datos, es mucha magia pero funciona
files = glob.glob('*/*/*.csv')
datalabels = {f : f.split("\\") for f in files}

spectredata = pd.concat([pd.read_csv(file, sep = ';', header = 1, decimal = ',', usecols = [0, 1])\
                             .assign(Group = labels[0],Sample = labels[1] ,File = labels[2])\
                        for file,labels in datalabels.items()], ignore_index=True)\
                .pipe(pd.pivot_table, values="Counts", index=["Group","Sample","File"],columns=["Wavelength [nm]"])
    
#acceso a datos se realiza escribiendo "spectredata.at[('12_02', 'M5', 'M5_20.csv'),293.83]" por ejemplo

#normalizamos los datos
scaler=StandardScaler()
scaler.fit(spectredata)
spectres_scaled=scaler.transform(spectredata)
     
#Instanciamos objeto PCA y aplicamos
pca=PCA(n_components=7) 
pca.fit(spectres_scaled)
spectres_pca=pca.transform(spectres_scaled) # convertimos nuestros datos con las nuevas dimensiones de PCA

#Por desgracia scikit devuelve arrays, así que creo un dataframe con los resultados del PCA y los índices de los datos originales
df_pca =  pd.DataFrame(data = spectres_pca, columns=["PCA" + str(i) for i in range(1,8)])
df_pca.index = spectredata.index

#GRÁFICOS
#gráficos aprovechando groupby de pandas

groups = df_pca.groupby(["Group"])#agrupa por compuesto

#Lo único horrible de este código para graficar es que no es explícito por qué cada grupo tiene un color diferente
#pero funciona; alguna alternativa sería trabajar con seaborn pero no sé bien como interactua con índices

fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.PCA1, group.PCA2, marker='o', linestyle='', ms=8, label=name)
ax.legend(numpoints=1, loc='upper left')
explained = 100 * pca.explained_variance_ratio_
ax.set_xlabel("PCA1 %2.2f %%"%explained[0])
ax.set_ylabel("PCA2 %2.2f %%"%explained[1])
ax.set_title("PCA archivos .spc")

#solución alternativa, usar seaborn (creo que es mejor)

pcapalette = ['blue'  , 'orange' , 'red' , 'green']
pcamarkers = ['o' , '^' , 's' , 'v']

df2 = df_pca.reset_index(level = 0) #esta línea convierte uno de los índices en una columna porque mucho código funciona más fácil así
ax = sns.scatterplot(data = df2, x = "PCA1", y = "PCA2", hue = 'Group', palette = pcapalette, style = 'Group', markers = pcamarkers)
ax.set_title("PCA archivos .spc")
ax.set_xlabel("PCA1 %2.2f %%"%explained[0])
ax.set_ylabel("PCA1 %2.2f %%"%explained[1])