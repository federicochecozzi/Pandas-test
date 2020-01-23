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
    return df[var][abs(df["Wavelength [nm]"] - w).idxmin()]

def integratebetween(df,wa,wb):
    return integrate.trapz(y = df["NormCounts"     ][df["Wavelength [nm]"].between(wa,wb)],
                           x = df["Wavelength [nm]"][df["Wavelength [nm]"].between(wa,wb)])
    
#PROCESAMIENTO DE DATOS
#Carga de datos
files = glob.glob('*/*/*.csv')#el resultado es de la forma "grupo\\muestra\\archivo"
datalabels = {f : f.split("\\") for f in files}#el diccionario es de la forma {"caminoarchivo":["grupo","muestra","archivo"]}

spectrefeatures = pd.concat(
                        #carga todos los archivos, los etiqueta adecuadamente y los une en un dataframe
                        #las columnas leídas de los archivos se llaman "Counts" y "Wavelength [nm]"
                        [pd.read_csv(file, sep = ';', header = 1, decimal = ',', usecols = [0, 1])
                         .assign(Group = labels[0],Sample = labels[1] ,File = labels[2]) 
                         for file,labels in datalabels.items()]
                        , ignore_index=True
                ).assign(
                    NormCounts = lambda df: df["Counts"] / valueatw(df,w = 325,var = "Counts")#columna con intensidades normalizadas 
                ).groupby(
                        ["Group","Sample","File"]#agrupo por archivo, preservando los otros índices
                ).apply(
                        lambda g: 
                                pd.Series(
                                            #uso pd.Series en vez de pd.DataFrame porque no es legal un dataframe de una sola fila, 
                                            #Pandas luego es suficientemente inteligente para unir las series y formar un dataframe
                                            {
                                                "Peak1"     : valueatw(g,w = 300,var = "NormCounts"),
                                                "Peak2"     : valueatw(g,w = 350,var = "NormCounts"),
                                                "Peak3"     : valueatw(g,w = 400,var = "NormCounts"),
                                                "Peak4"     : valueatw(g,w = 450,var = "NormCounts"),
                                                "Integral1" : integratebetween(g,wa = 275,wb = 325),
                                                "Integral2" : integratebetween(g,wa = 325,wb = 375),
                                                "Integral3" : integratebetween(g,wa = 375,wb = 425),
                                                "Integral4" : integratebetween(g,wa = 425,wb = 475)
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