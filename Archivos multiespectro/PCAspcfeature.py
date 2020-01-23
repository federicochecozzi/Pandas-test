# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:06:36 2019

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
from scipy import integrate

def valueatw(df,w,var):
    return df[var][abs(df["Wavelength"] - w).idxmin()]

def integratebetween(df,wa,wb):
    return integrate.trapz(y = df["NormIntensity"][df["Wavelength"].between(wa,wb)],
                           x = df["Wavelength"   ][df["Wavelength"].between(wa,wb)])

def normalize(df,w):
    df["NormIntensity"] = df["Intensity"] / valueatw(df,w = 481.8,var = "Intensity")#columna con intensidades normalizadas 
    return df
    
#PROCESAMIENTO DE DATOS
#Carga de datos
files = glob.glob('*.csv')

spectrefeatures = pd.concat(
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
                                ["Group","Sample","Measurement"]#agrupo por archivo, preservando los otros índices
                            ).apply(
                                normalize, w = 481.8#normalización por grupo 
                            ).groupby(
                                ["Group","Sample","Measurement"]
                            ).apply(
                                lambda g: 
                                pd.Series(
                                            #uso pd.Series en vez de pd.DataFrame porque no es legal un dataframe de una sola fila, 
                                            #Pandas luego es suficientemente inteligente para unir las series y formar un dataframe
                                            {
                                                "Peak1"     : valueatw(g,w = 472.6 ,var = "NormIntensity"),
                                                "Peak2"     : valueatw(g,w = 492.99,var = "NormIntensity"),
                                                "Peak3"     : valueatw(g,w = 500.7 ,var = "NormIntensity"),
                                                "Peak4"     : valueatw(g,w = 512.24,var = "NormIntensity"),
                                                "Integral1" : integratebetween(g,wa = 471,wb = 474.5),
                                                "Integral2" : integratebetween(g,wa = 488,wb = 497),
                                                "Integral3" : integratebetween(g,wa = 498,wb = 506),
                                                "Integral4" : integratebetween(g,wa = 509,wb = 515)
                                            }
                                        )#extraigo las características a usar en el análisis de datos de cada grupo
                            )
                        
#Normalizamos los datos
scaler=StandardScaler()
scaler.fit(spectrefeatures)
spectres_scaled=scaler.transform(spectrefeatures)
     
#Instanciamos objeto PCA y aplicamos
pca=PCA(n_components=7) 
pca.fit(spectres_scaled)
spectres_pca=pca.transform(spectres_scaled) # convertimos nuestros datos con las nuevas dimensiones de PCA
explained = 100 * pca.explained_variance_ratio_

#Por desgracia scikit devuelve arrays, así que creo un dataframe con los resultados del PCA y los índices de los datos originales
df_pca =  pd.DataFrame(data = spectres_pca, columns=["PCA" + str(i) for i in range(1,8)])
df_pca.index = spectrefeatures.index

#GRÁFICOS

pcapalette = ['blue'  , 'orange' , 'red' , 'green']
pcamarkers = ['o' , '^' , 's' , 'v']

df_pca = df_pca.reset_index(level = 0) #esta línea convierte el índice con los grupos porque seaborn trabaja con columnas
ax = sns.scatterplot(data = df_pca, x = "PCA1", y = "PCA2", hue = 'Group', palette = pcapalette, style = 'Group', markers = pcamarkers)
ax.set_title("PCA archivos .spc")
ax.set_xlabel("PCA1 %2.2f %%"%explained[0])
ax.set_ylabel("PCA1 %2.2f %%"%explained[1])