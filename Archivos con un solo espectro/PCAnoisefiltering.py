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
#import matplotlib.pyplot as plt
import seaborn as sns

def threshold(df, percentofmax):
    lowerbound = df.Intensity.max() * percentofmax / 100
    df.Intensity[df.Intensity <= lowerbound] = 0
    return df#.assign( FilteredIntensity = lambda df: df.Intensity.clip(lower = lowerbound))     

#PROCESAMIENTO DE DATOS
#carga de datos, es mucha magia pero funciona
files = glob.glob('*/*/*.csv')
datalabels = {f : f.split("\\") for f in files}

percentofmax = 5
                
spectredata = pd.concat(
                            [pd.read_csv(file, sep = ';', header = 1, decimal = ',', usecols = [0, 1], names = ["Wavelength","Intensity"])
                             .assign(Group = labels[0],Sample = labels[1] ,Measurement = labels[2].rstrip(".csv"))
                             for file,labels in datalabels.items()], ignore_index=True
                        ).groupby(
                            ["Group","Sample","Measurement"]
                        ).apply(
                            threshold, percentofmax = percentofmax
                        ).pivot_table(
                            values="Intensity", index=["Group","Sample","Measurement"],columns=["Wavelength"]
                        )
    
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

explained = 100 * pca.explained_variance_ratio_

pcapalette = ['blue'  , 'orange' , 'red' , 'green']
pcamarkers = ['o' , '^' , 's' , 'v']

df2 = df_pca.reset_index(level = 0) #esta línea convierte uno de los índices en una columna porque mucho código funciona más fácil así
ax = sns.scatterplot(data = df2, x = "PCA1", y = "PCA2", hue = 'Group', palette = pcapalette, style = 'Group', markers = pcamarkers)
ax.set_title("PCA archivos .spc filtrado para %d %% del máximo"%percentofmax)
ax.set_xlabel("PCA1 %2.2f %%"%explained[0])
ax.set_ylabel("PCA2 %2.2f %%"%explained[1])