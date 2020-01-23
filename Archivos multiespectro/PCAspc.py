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
import seaborn as sns

def valueatw(df,w,var):
    return df[var][abs(df["Wavelength"] - w).idxmin()]

def normalize(df,w):
    df["NormIntensity"] = df["Intensity"] / valueatw(df,w = 481.8,var = "Intensity")#columna con intensidades normalizadas 
    return df
    
#PROCESAMIENTO DE DATOS
#carga de datos, es mucha magia pero funciona
files = glob.glob('*.csv')

spectredata = pd.concat(
                            [pd.read_csv(f, sep = ';', header = 1, decimal = ',', usecols = list(range(0,12)),names =["Wavelength","Measure1","Measure2","Measure3","Measure4","Measure5","Measure6","Measure7","Measure8","Measure9","Measure10","Measure11"])
                             .assign(Sample = f.rstrip(".csv"), Group = f.rstrip(".csv").rstrip("0123456789"))
                                 for f in files], ignore_index=True
                        ).melt(
                            id_vars=["Group","Sample","Wavelength"], value_vars =["Measure1","Measure2","Measure3","Measure4","Measure5","Measure6","Measure7","Measure8","Measure9","Measure10","Measure11"],
                            var_name = "Measurement", value_name = "Intensity"
                        ).query(
                            "Wavelength >= 465 & Wavelength <=520" 
                        ).query(
                            "(Sample == ['PE1','PE2'] & Measurement == ['Measure3','Measure4','Measure5']) | (Sample != ['PE1','PE2'] & Measurement == ['Measure2','Measure3','Measure4'])"
                        ).groupby(
                            ["Group","Sample","Measurement"]#agrupo por medición, preservando los otros índices
                        ).apply(
                            normalize, w = 481.8#normalización por grupo 
                        ).pivot_table(
                            values = "NormIntensity", index=["Group","Sample","Measurement"], columns = "Wavelength"
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
ax.set_title("PCA archivos .spc")
ax.set_xlabel("PCA1 %2.2f %%"%explained[0])
ax.set_ylabel("PCA1 %2.2f %%"%explained[1])