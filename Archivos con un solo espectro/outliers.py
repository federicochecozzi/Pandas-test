# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:51:26 2020

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

#IMPORTANTE
#Usar F9 para correr bloques de código seleccionados; eso ahorra tener que correr el script entero cada vez que se necesita probar algo

#Primera serie de gráficos: detectar señales con mal aspecto (visualización por grupo)
files = glob.glob('*/*/*.csv')
datalabels = {f : f.split("\\") for f in files}

spectredata = pd.concat(
                            [pd.read_csv(file, sep = ';', header = 1, decimal = ',', usecols = [0, 1], names = ["Wavelength","Intensity"])
                             .assign(Group = labels[0],Sample = labels[1] ,Measurement = labels[2])
                                 for file,labels in datalabels.items()], ignore_index=True
                        )

groups = spectredata.groupby(["Sample"])

for name, group in groups:
    plt.figure()
    sns.lineplot(data = group, x = "Wavelength", y = "Intensity", hue = "Measurement", legend = False)
    
#Segunda serie de gráficos: PCA con ciertos arreglos cosméticos
spectredata = pd.concat(
                            [pd.read_csv(file, sep = ';', header = 1, decimal = ',', usecols = [0, 1], names = ["Wavelength","Intensity"])
                             .assign(Group = labels[0],Sample = labels[1] ,Measurement = labels[2].rstrip(".csv"))
                             for file,labels in datalabels.items()], ignore_index=True
                        ).pivot_table(
                            values="Intensity", index=["Group","Sample","Measurement"],columns=["Wavelength"]
                        ).query(
                            "Group != '05_01'"  #filtro uno de los grupos  
                        ).query(#esto es para remover los outliers antes de realizar el PCA, en base a los resultados del PCA previo
                            "Measurement != ['M13_01','M13_12','M13_16','M13_11','M13_14','M5_10','M5_09','M5_13']"
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

pcapalette = ['blue'  , 'orange' , 'red']# , 'green']
pcamarkers = ['o' , '^' , 's' ]#, 'v']

df2 = df_pca.reset_index(level = 0) #esta línea convierte uno de los índices en una columna porque mucho código funciona más fácil así
ax = sns.scatterplot(data = df2, x = "PCA1", y = "PCA2", hue = 'Group', palette = pcapalette, style = 'Group', markers = pcamarkers)
ax.set_title("PCA archivos .spc")
ax.set_xlabel("PCA1 %2.2f %%"%explained[0])
ax.set_ylabel("PCA1 %2.2f %%"%explained[1])
for index,row in df2.iterrows():#etiquetas de punto
    ax.annotate(index[1], row[1:3],size = 12)#Dios santo esto es muy feo
    
#gráfico sin los outliers 1
dfclean = df2.query("Measurement != ['M13_01','M13_12','M13_16','M13_11','M13_14','M5_10','M5_09','M5_13']")
ax = sns.scatterplot(data = dfclean, x = "PCA1", y = "PCA2", hue = 'Group', palette = pcapalette, style = 'Group', markers = pcamarkers)
ax.set_title("PCA archivos .spc con outliers removidos")
ax.set_xlabel("PCA1 %2.2f %%"%explained[0])
ax.set_ylabel("PCA1 %2.2f %%"%explained[1])
    