# Cargo las librerias 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import itertools
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, 
                           atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali, lm_custom, 
                           lm_stepwise, glm_stepwise, validacion_cruzada_glm, lm_forward, lm_backward, glm_forward, glm_backward, glm)

random.seed(42)

# Cargo los datos
datos = pd.read_excel('src/data/DatosEleccionesEspana.xlsx')

# Eliminamos las variables que no usaremos
variables_a_eliminar = ["Izda_Pct", "Dcha_Pct", "Otros_Pct", "Izquierda", "Derecha"]

datos = datos.drop(columns=variables_a_eliminar)

# Comprobamos el tipo de formato de las variables variable que se ha asignado en la lectura.
print(datos.dtypes)

# Dimensiones del dataframe
print(datos.shape)

# Genera una lista con los nombres de las variables.
variables = list(datos.columns)  

# Seleccionar las columnas numéricas del DataFrame
numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Seleccionar las columnas categóricas del DataFrame
categoricas = [variable for variable in variables if variable not in numericas]

print(f"Numericas: {len(numericas)}, Categoricas: {len(categoricas)}")

# Frecuencias de los valores en las variables categóricas
analisis_categoricas = analizar_variables_categoricas(datos)

print(analisis_categoricas)

# Cuenta el número de valores distintos de cada una de las variables numéricas de un DataFrame
print(cuentaDistintos(datos))

# Descriptivos variables numéricas mediante función describe() de Python
descriptivos_num = datos.describe().T

print(descriptivos_num)
# Añadimos más descriptivos a los anteriores
for num in numericas:
    descriptivos_num.loc[num, "Asimetria"] = datos[num].skew()
    descriptivos_num.loc[num, "Kurtosis"] = datos[num].kurtosis()
    descriptivos_num.loc[num, "Rango"] = np.ptp(datos[num].dropna().values)

print(descriptivos_num)


# Corregimos los errores detectados

# A veces los 'nan' vienen como como una cadena de caracteres, los modificamos a perdidos.
for x in categoricas:
    datos[x] = datos[x].replace('nan', np.nan) 

# Missings no declarados variables cualitativas (NSNC, ?)
datos['Densidad'] = datos['Densidad'].replace('?', np.nan)

# Missings no declarados variables cuantitativas (-1, 99999)
datos['Explotaciones'] = datos['Explotaciones'].replace(99999, np.nan)

# Valores fuera de rango
datos['ForeignersPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['ForeignersPtge']]
datos['SameComAutonPtge'] = [x if 0 <= x <= 100 else np.nan for x in datos['SameComAutonPtge']]
datos['PobChange_pct'] = [x if x <= 100 else np.nan for x in datos['PobChange_pct']]

# Cambiamos tipo de la binaria a int
datos['AbstencionAlta'] = datos['AbstencionAlta'].astype(int)

# Indico la variableObj, el ID y las Input (los atipicos y los missings se gestionan
# solo de las variables input)
datos = datos.set_index(datos['Name']).drop('Name', axis = 1)
varObjCont = datos['AbstentionPtge']
varObjBin = datos['AbstencionAlta']
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1)

# Genera una lista con los nombres de las variables del cojunto de datos input.
variables_input = list(datos_input.columns)  

# Selecionamos las variables numéricas
numericas_input = datos_input.select_dtypes(include = ['int', 'int32', 'int64','float', 'float32', 'float64']).columns

# Selecionamos las variables categóricas
categoricas_input = [variable for variable in variables_input if variable not in numericas_input]


## ATIPICOS

# Cuento el porcentaje de atipicos de cada variable. 
resultados = {x: atipicosAmissing(datos_input[x])[1] / len(datos_input) for x in numericas_input}

print(resultados)

# Modifico los atipicos como missings
for x in numericas_input:
    datos_input[x] = atipicosAmissing(datos_input[x])[0]

# MISSINGS
    
# Muestra valores perdidos
variables = list(datos.columns)  
print(datos[variables].isna().sum())

# Visualiza un mapa de calor que muestra la matriz de correlación de valores ausentes en el conjunto de datos.
patron_perdidos(datos_input)

# Muestra total de valores perdidos por cada variable
print(datos_input[variables_input].isna().sum())

# Muestra proporción de valores perdidos por cada variable (guardo la información)
prop_missingsVars = datos_input.isna().sum()/len(datos_input)

print(prop_missingsVars)

# Creamos la variable prop_missings que recoge el número de valores perdidos por cada observación
datos_input['prop_missings'] = datos_input.isna().mean(axis = 1)
print(datos_input)

# Realizamos un estudio descriptivo básico a la nueva variable
print(datos_input['prop_missings'].describe())

# Calculamos el número de valores distintos que tiene la nueva variable
print(len(datos_input['prop_missings'].unique()))

# Elimino las observaciones con mas de la mitad de datos missings (no hay ninguna)
eliminar = datos_input['prop_missings'] > 0.5
datos_input = datos_input[~eliminar]
varObjBin = varObjBin[~eliminar]
varObjCont = varObjCont[~eliminar]

# Transformo la nueva variable en categórica (ya que tiene pocos valores diferentes)
datos_input["prop_missings"] = datos_input["prop_missings"].astype(str)

# Agrego 'prop_missings' a la lista de nombres de variables input
variables_input.append('prop_missings')
categoricas_input.append('prop_missings')


# Elimino las variables con mas de la mitad de datos missings (no hay ninguna)
eliminar = [prop_missingsVars.index[x] for x in range(len(prop_missingsVars)) if prop_missingsVars[x] > 0.5]
datos_input = datos_input.drop(eliminar, axis = 1)

## IMPUTACIONES
# Imputo todas las cuantitativas, seleccionar el tipo de imputacion: media, mediana o aleatorio
for x in numericas_input:
    simetria = datos_input[x].skew()
    if simetria < 1:
        datos_input[x] = ImputacionCuant(datos_input[x], 'media')
    else:
        datos_input[x] = ImputacionCuant(datos_input[x], 'mediana')

# Imputo todas las cualitativas, seleccionar el tipo de imputacion: moda o aleatorio
for x in categoricas_input:
    datos_input[x] = ImputacionCuali(datos_input[x], 'moda')

# Reviso que no queden datos missings
print(datos_input.isna().sum())


# Una vez finalizado este proceso, se puede considerar que los datos estan depurados. Los guardamos
datosElecciones = pd.concat([varObjBin, varObjCont, datos_input], axis = 1)
with open('datosEleccionesDep.pickle', 'wb') as archivo:
    pickle.dump(datosElecciones, archivo)


# Cargo las funciones que voy a utilizar despues
from FuncionesMineria import *

# Parto de los datos ya depurados
with open('datosEleccionesDep.pickle', 'rb') as f:
    datos = pickle.load(f)

# Defino las variables objetivo y las elimino del conjunto de datos input
varObjCont = datos['AbstentionPtge']
varObjBin = datos['AbstencionAlta']
datos_input = datos.drop(['AbstentionPtge', 'AbstencionAlta'], axis = 1) 


# Como vemos que las escalas entre variables difieren mucho, vamos a normalizar las variables para que todas estén en escala 0-1
# Esto ayuda a la regresión tanto logística como lineal
print(datos_input.describe().T)

# Separate numeric columns for normalization
numeric_columns = datos_input.select_dtypes(include=['float64', 'int64', 'int32', 'float32']).columns

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the numeric columns
datos_input[numeric_columns] = scaler.fit_transform(datos_input[numeric_columns])

# Vemos ahora que todos los datos numéricos están normalizados
print(datos_input.describe().T)

# Obtengo la importancia de las variables Categóricas
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)


# Veo graficamente el efecto de dos variables cualitativas sobre la binaria
# Tomo las variables con más y menos relación con la variable objetivo Binaria
mosaico_targetbinaria(datos_input['Densidad'], varObjBin, 'Densidad')
mosaico_targetbinaria(datos_input['CCAA'], varObjBin, 'Comunidades Autonomas')

# Veo graficamente el efecto de dos variables cuantitativas sobre la binaria
boxplot_targetbinaria(datos_input['Population'], varObjBin, 'Población')
boxplot_targetbinaria(datos_input['Age_over65_pct'], varObjBin, 'Porcentaje mayores 65 años')

hist_targetbinaria(datos_input['Population'], varObjBin, 'Población')
hist_targetbinaria(datos_input['Age_over65_pct'], varObjBin, 'Porcentaje mayores 65 años')

# Correlación entre todas las variables numéricas frente a la objetivo continua.
# Obtener las columnas numéricas del DataFrame 'datos_input'
numericas = datos_input.select_dtypes(include=['int', 'float']).columns
# Calcular la matriz de correlación de Pearson entre la variable objetivo continua ('varObjCont') y las variables numéricas
matriz_corr = pd.concat([varObjCont, datos_input[numericas]], axis = 1).corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
plt.figure(figsize=(10, 8))
# Crea un mapa de calor utilizando la máscara
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Establecer el título del gráfico
plt.title("Matriz de correlación")
# Mostrar el gráfico de la matriz de correlación
plt.show()

# Eliminamos variables altamente correlacionadas
columns_to_remove = ["TotalCensus", "Pob2010", "inmuebles", "Servicios", "ComercTTEHosteleria", 
                     "Construccion", "Industria", 
                     "totalEmpresas"]

datos_input = datos_input.drop(columns=columns_to_remove)

# Eliminamos variables redundantes de porcentajes
columns_to_remove = ["Age_0-4_Ptge"]

datos_input = datos_input.drop(columns=columns_to_remove)

# Recategorizo uniendo Ceuta, Melilla y Murcia por tener una actividad principal similar (Hostelería, servicios u otros)
# y por tener esto influencia notable sobre la variable objetivo, como vemos más adelante
datos_input['CCAA'] = datos_input['CCAA'].replace({'Ceuta': 'Murcia_Ceuta_Melilla','Melilla': 'Murcia_Ceuta_Melilla', 'Murcia': 'Murcia_Ceuta_Melilla'})

# Uno industria y construcción porque las ciudades en las que predominan suelen tener la misma tendencia electoral
datos_input['ActividadPpal'] = datos_input['ActividadPpal'].replace({'Construccion': 'Industria_Construccion','Industria': 'Industria_Construccion'})

numericas = datos_input.select_dtypes(include=['int', 'float']).columns

# Busco las mejores transformaciones para las variables numericas con respesto a los dos tipos de variables
print(Transf_Auto(datos_input[numericas], varObjCont))

input_cont = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjCont)], axis = 1)
input_bin = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjBin)], axis = 1)


# Creamos conjuntos de datos que contengan las variables explicativas y una de las variables objetivo y los guardamos
todo_cont = pd.concat([input_cont, varObjCont], axis = 1)
todo_bin = pd.concat([input_bin, varObjBin], axis = 1)

# Calcular la matriz de correlación de Pearson entre la variable objetivo continua ('varObjCont') y las variables numéricas
matriz_corr = todo_cont.corr(method = 'pearson')
# Crear una máscara para ocultar la mitad superior de la matriz de correlación (triangular superior)
mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
# Crear una figura para el gráfico con un tamaño de 8x6 pulgadas
plt.figure(figsize=(10, 8))
# Crea un mapa de calor utilizando la máscara
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Establecer el título del gráfico
plt.title("Matriz de correlación")
# Mostrar el gráfico de la matriz de correlación
plt.show()

with open('todo_bin.pickle', 'wb') as archivo:
    pickle.dump(todo_bin, archivo)
with open('todo_cont.pickle', 'wb') as archivo:
    pickle.dump(todo_cont, archivo)


## Comenzamos con la regresion lineal
# Hago de nuevo la partición porque hay una nueva variable en el conjunto de datos "Todo"
x_train, x_test, y_train, y_test = train_test_split(input_cont, varObjCont, test_size = 0.2, random_state = 1234567)

# Genera una lista con los nombres de las variables.
variables = list(input_cont.columns)  

# Variables numéricas originales
var_cont = numericas.to_list()

# Variables categ originales
var_categ = [x for x in categoricas_input if x !="prop_missings"]

# Seleccionar las columnas numéricas del DataFrame
var_cont_con_transf = input_cont.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns.to_list()

# Interacciones 2 a 2 de todas las variables (excepto las continuas transformadas)
interacciones = var_categ
interacciones_unicas = list(itertools.combinations(interacciones, 2))


# MODELO 0 Selección aleatoria
## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)
# Concretamente el 70% de los datos de entrenamiento utilizados para contruir los
# modelos anteriores.
# El método de selección usado ha sido el Stepwise con el criterio BIC
# Se aplica este método a 30 submuestras diferentes

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
   
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train,
                                                            test_size = 0.3, random_state = 1234567 + x)
   
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = lm_stepwise(y_train2, x_train2, var_cont_con_transf, var_categ, interacciones_unicas, 'BIC')
   
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append(sorted(modelo['Modelo'].model.exog_names))

# Unir las variables en las fórmulas seleccionadas en una sola cadena.
variables_seleccionadas['Formula'] = list(map(lambda x: '+'.join(x), variables_seleccionadas['Formula']))
   
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia.
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns = ['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending = False).reset_index()

# Identificar las dos modelos más frecuentes y las variables correspondientes.
ivar_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]
var_2 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][1])]

# Con las variables obtenidas entrenamos el modelo aleatorio

modelo_aleatorio = lm(y_train, x_train, ivar_1['cont'], ivar_1['categ'],
                                 ivar_1['inter'])

# Resumen del modelo
print(modelo_aleatorio['Modelo'].summary())

# R-squared del modelo para train
print(Rsq(modelo_aleatorio['Modelo'], y_train, modelo_aleatorio['X']))

# Preparo datos test
x_test_aleatorio = crear_data_modelo(x_test, modelo_aleatorio['Variables']['cont'], 
                                                    modelo_aleatorio['Variables']['categ'], 
                                                    modelo_aleatorio['Variables']['inter'])
# R-squared del modelo para test
print(Rsq(modelo_aleatorio['Modelo'], y_test, x_test_aleatorio))


# MODELO 1 forward, métrica AIC  con transformaciones con interacciones

modeloForwardAIC_con_trans_con_int = lm_forward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'AIC')

# Resumen del modelo
print(modeloForwardAIC_con_trans_con_int['Modelo'].summary())

# R-squared del modelo para train
print(Rsq(modeloForwardAIC_con_trans_con_int['Modelo'], y_train, modeloForwardAIC_con_trans_con_int['X']))

# Preparo datos test
x_test_modeloForwardAIC_con_trans_con_int = crear_data_modelo(x_test, modeloForwardAIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloForwardAIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloForwardAIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(Rsq(modeloForwardAIC_con_trans_con_int['Modelo'], y_test, x_test_modeloForwardAIC_con_trans_con_int))


# MODELO 2 backward, métrica AIC  con transformaciones con interacciones

modeloBackwardAIC_con_trans_con_int = lm_backward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'AIC')

# Resumen del modelo
print(modeloBackwardAIC_con_trans_con_int['Modelo'].summary())

# R-squared del modelo para train
print(Rsq(modeloBackwardAIC_con_trans_con_int['Modelo'], y_train, modeloBackwardAIC_con_trans_con_int['X']))

# Preparo datos test
x_test_modeloBackwardAIC_con_trans_con_int = crear_data_modelo(x_test, modeloBackwardAIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloBackwardAIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloBackwardAIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(Rsq(modeloBackwardAIC_con_trans_con_int['Modelo'], y_test, x_test_modeloBackwardAIC_con_trans_con_int))

# MODELO 3 Stepwise, métrica AIC  con transformaciones con interacciones

modeloStepAIC_con_trans_con_int = lm_stepwise(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'AIC')

# Resumen del modelo
print(modeloStepAIC_con_trans_con_int['Modelo'].summary())

# R-squared del modelo para train
print(Rsq(modeloStepAIC_con_trans_con_int['Modelo'], y_train, modeloStepAIC_con_trans_con_int['X']))

# Preparo datos test
x_test_modeloStepAIC_con_trans_con_int = crear_data_modelo(x_test, modeloStepAIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloStepAIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloStepAIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(Rsq(modeloStepAIC_con_trans_con_int['Modelo'], y_test, x_test_modeloStepAIC_con_trans_con_int))


# MODELO 4 forward, métrica BIC  con transformaciones con interacciones
modeloForwardBIC_con_trans_con_int = lm_forward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'BIC')

# Resumen del modelo
print(modeloForwardBIC_con_trans_con_int['Modelo'].summary())

# R-squared del modelo para train
print(Rsq(modeloForwardBIC_con_trans_con_int['Modelo'], y_train, modeloForwardBIC_con_trans_con_int['X']))

# Preparo datos test
x_test_modeloForwardBIC_con_trans_con_int = crear_data_modelo(x_test, modeloForwardBIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloForwardBIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloForwardBIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(Rsq(modeloForwardBIC_con_trans_con_int['Modelo'], y_test, x_test_modeloForwardBIC_con_trans_con_int))

# MODELO 5 backward, métrica BIC  con transformaciones con interacciones
modeloBackwardBIC_con_trans_con_int = lm_backward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'BIC')

# Resumen del modelo
print(modeloBackwardBIC_con_trans_con_int['Modelo'].summary())

# R-squared del modelo para train
print(Rsq(modeloBackwardBIC_con_trans_con_int['Modelo'], y_train, modeloBackwardBIC_con_trans_con_int['X']))

# Preparo datos test
x_test_modeloBackwardBIC_con_trans_con_int = crear_data_modelo(x_test, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloBackwardBIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloBackwardBIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(Rsq(modeloBackwardBIC_con_trans_con_int['Modelo'], y_test, x_test_modeloBackwardBIC_con_trans_con_int))

# MODELO 6 Stepwise, métrica BIC  con transformaciones con interacciones
modeloStepBIC_con_trans_con_int = lm_stepwise(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'BIC')

# Resumen del modelo
print(modeloStepBIC_con_trans_con_int['Modelo'].summary())

# R-squared del modelo para train
print(Rsq(modeloStepBIC_con_trans_con_int['Modelo'], y_train, modeloStepBIC_con_trans_con_int['X']))

# Preparo datos test
x_test_modeloStepBIC_con_trans_con_int = crear_data_modelo(x_test, modeloStepBIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloStepBIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloStepBIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(Rsq(modeloStepBIC_con_trans_con_int['Modelo'], y_test, x_test_modeloStepBIC_con_trans_con_int))

#########################################################################################################################################
########################################### TABLA RESUMEN ###############################################################################
# Suponiendo que tienes estas listas con la información necesaria
modelos = ['Modelo aleatorio', 'Modelo Backward AIC', 'Modelo Forward AIC', 'Modelo Stepwise AIC', 'Modelo Backward BIC', 'Modelo Forward BIC', 'Modelo Stepwise BIC']

num_parametros = [
                    len(modelo_aleatorio['Modelo'].params),
                    len(modeloBackwardAIC_con_trans_con_int['Modelo'].params), 
                    len(modeloForwardAIC_con_trans_con_int['Modelo'].params), 
                    len(modeloStepAIC_con_trans_con_int['Modelo'].params), 
                    len(modeloBackwardBIC_con_trans_con_int['Modelo'].params), 
                    len(modeloForwardBIC_con_trans_con_int['Modelo'].params), 
                    len(modeloStepBIC_con_trans_con_int['Modelo'].params)
                ]

r2_test = [
            Rsq(modelo_aleatorio['Modelo'], y_test, x_test_aleatorio),
            Rsq(modeloBackwardAIC_con_trans_con_int['Modelo'], y_test, x_test_modeloBackwardAIC_con_trans_con_int), 
            Rsq(modeloForwardAIC_con_trans_con_int['Modelo'], y_test, x_test_modeloForwardAIC_con_trans_con_int),
            Rsq(modeloStepAIC_con_trans_con_int['Modelo'], y_test, x_test_modeloStepAIC_con_trans_con_int),
            Rsq(modeloBackwardBIC_con_trans_con_int['Modelo'], y_test, x_test_modeloBackwardBIC_con_trans_con_int), 
            Rsq(modeloForwardBIC_con_trans_con_int['Modelo'], y_test, x_test_modeloForwardBIC_con_trans_con_int),
            Rsq(modeloStepBIC_con_trans_con_int['Modelo'], y_test, x_test_modeloStepBIC_con_trans_con_int)
            ]

r2_train = [
            Rsq(modelo_aleatorio['Modelo'], y_train, modelo_aleatorio['X']),
            Rsq(modeloBackwardAIC_con_trans_con_int['Modelo'], y_train, modeloBackwardAIC_con_trans_con_int['X']),
            Rsq(modeloForwardAIC_con_trans_con_int['Modelo'], y_train, modeloForwardAIC_con_trans_con_int['X']),
            Rsq(modeloStepAIC_con_trans_con_int['Modelo'], y_train, modeloStepAIC_con_trans_con_int['X']),
            Rsq(modeloBackwardBIC_con_trans_con_int['Modelo'], y_train, modeloBackwardBIC_con_trans_con_int['X']),
            Rsq(modeloForwardBIC_con_trans_con_int['Modelo'], y_train, modeloForwardBIC_con_trans_con_int['X']),
            Rsq(modeloStepBIC_con_trans_con_int['Modelo'], y_train, modeloStepBIC_con_trans_con_int['X'])
            ]

# Crear el DataFrame
df = pd.DataFrame({
    'Modelo': modelos,
    'Número de Parámetros': num_parametros,
    'R2 Train': r2_train,
    'R2 Test': r2_test
})

# Mostrar el DataFrame
print(df)

df.to_csv('tabla_resultados_regresion_lineal.csv')

#########################################################################################################################################


# Hago validacion cruzada repetida para ver que modelo es mejor
# Los mejores son: Stepwise AIC y BIC y Forward BIC
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en seis modelos diferentes y almacena sus R-squared en listas separadas

    modelo_stepBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloStepBIC_con_trans_con_int['Variables']['cont']
        , modeloStepBIC_con_trans_con_int['Variables']['categ']
        , modeloStepBIC_con_trans_con_int['Variables']['inter']
    )
    modelo_forwardBIC = validacion_cruzada_lm(
        5
        , x_train
        , y_train
        , modeloForwardBIC_con_trans_con_int['Variables']['cont']
        , modeloForwardBIC_con_trans_con_int['Variables']['categ']
        , modeloForwardBIC_con_trans_con_int['Variables']['inter']
    )

    results_rep = pd.DataFrame({
        'Rsquared': modelo_stepBIC + modelo_forwardBIC
        , 'Resample': ['Rep' + str((rep + 1))]*5*2 # Etiqueta de repetición (5 repeticiones 2 modelos)
        , 'Modelo': [1]*5 + [2]*5 # Etiqueta de modelo (3 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)
    
# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de Rsquared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico 


results.to_csv("resultados_validacion_cruzada_reg_lineal.csv")
print(results)



## Modelo ganador: Stepwise BIC
print(modeloStepBIC_con_trans_con_int['Modelo'].summary())



#######################################################################################################
#################################### R. Logistica #####################################################
x_train, x_test, y_train, y_test = train_test_split(input_bin, varObjBin, test_size = 0.2, random_state = 1234567)

# Genera una lista con los nombres de las variables.
variables = list(input_bin.columns)  

# Variables numéricas originales
var_cont = numericas.to_list()

# Variables categ originales
var_categ = [x for x in categoricas_input if x !="prop_missings"]

# Seleccionar las columnas numéricas del DataFrame
var_cont_con_transf = input_bin.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64']).columns.to_list()

# Interacciones 2 a 2 de todas las variables (excepto las continuas transformadas)
interacciones = var_categ
interacciones_unicas = list(itertools.combinations(interacciones, 2))


# MODELO 0 Selección aleatoria
## Seleccion aleatoria (se coge la submuestra de los datos de entrenamiento)
# Concretamente el 70% de los datos de entrenamiento utilizados para contruir los
# modelos anteriores.
# El método de selección usado ha sido el Stepwise con el criterio BIC
# Se aplica este método a 30 submuestras diferentes

# Inicializar un diccionario para almacenar las fórmulas y variables seleccionadas.
variables_seleccionadas = {
    'Formula': [],
    'Variables': []
}

# Realizar 30 iteraciones de selección aleatoria.
for x in range(30):
    print('---------------------------- iter: ' + str(x))
   
    # Dividir los datos de entrenamiento en conjuntos de entrenamiento y prueba.
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_train, y_train,
                                                            test_size = 0.3, random_state = 1234567 + x)
   
    # Realizar la selección stepwise utilizando el criterio BIC en la submuestra.
    modelo = glm_stepwise(y_train2, x_train2, var_cont_con_transf, var_categ, interacciones_unicas, 'BIC')
   
    # Almacenar las variables seleccionadas y la fórmula correspondiente.
    variables_seleccionadas['Variables'].append(modelo['Variables'])
    variables_seleccionadas['Formula'].append('+'.join(modelo['Variables']))
  
# Calcular la frecuencia de cada fórmula y ordenarlas por frecuencia
frecuencias = Counter(variables_seleccionadas['Formula'])
frec_ordenada = pd.DataFrame(list(frecuencias.items()), columns=['Formula', 'Frecuencia'])
frec_ordenada = frec_ordenada.sort_values('Frecuencia', ascending=False).reset_index(drop=True)

# Identificar las dos modelos más frecuentes y las variables correspondientes.
ivar_1 = variables_seleccionadas['Variables'][variables_seleccionadas['Formula'].index(
    frec_ordenada['Formula'][0])]

# Con las variables obtenidas entrenamos el modelo aleatorio
modelo_aleatorio = glm(y_train, x_train, ivar_1['cont'], ivar_1['categ'],
                                 ivar_1['inter'])

# Resumen del modelo
print(summary_glm(modelo_aleatorio['Modelo'], y_train, modelo_aleatorio['X']))

# R-squared del modelo para train
print(pseudoR2(modelo_aleatorio['Modelo'],  modelo_aleatorio['X'],  y_train))

# Preparo datos test
x_test_aleatorio = crear_data_modelo(x_test, modelo_aleatorio['Variables']['cont'], 
                                                    modelo_aleatorio['Variables']['categ'], 
                                                    modelo_aleatorio['Variables']['inter'])

# R-squared del modelo para test
print(pseudoR2(modelo_aleatorio['Modelo'], x_test_aleatorio,  y_test))

# MODELO 1 forward, métrica AIC  con transformaciones con interacciones
modeloForwardAIC_con_trans_con_int = glm_forward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'AIC')

# Resumen del modelo
print(summary_glm(modeloForwardAIC_con_trans_con_int['Modelo'], y_train, modeloForwardAIC_con_trans_con_int['X']))

# R-squared del modelo para train
print(pseudoR2(modeloForwardAIC_con_trans_con_int['Modelo'], modeloForwardAIC_con_trans_con_int['X'],  y_train))

# Preparo datos test
x_test_modeloForwardAIC_con_trans_con_int = crear_data_modelo(x_test, modeloForwardAIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloForwardAIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloForwardAIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(pseudoR2(modeloForwardAIC_con_trans_con_int['Modelo'],  x_test_modeloForwardAIC_con_trans_con_int, y_test))


# MODELO 2 backward, métrica AIC  con transformaciones con interacciones
modeloBackwardAIC_con_trans_con_int = glm_backward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'AIC')

# Resumen del modelo
print(summary_glm(modeloBackwardAIC_con_trans_con_int['Modelo'], y_train, modeloBackwardAIC_con_trans_con_int['X']))

# R-squared del modelo para train
print(pseudoR2(modeloBackwardAIC_con_trans_con_int['Modelo'], modeloBackwardAIC_con_trans_con_int['X'],  y_train))

# Preparo datos test
x_test_modeloBackwardAIC_con_trans_con_int = crear_data_modelo(x_test, modeloBackwardAIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloBackwardAIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloBackwardAIC_con_trans_con_int['Variables']['inter'])

# R-squared del modelo para test
print(pseudoR2(modeloBackwardAIC_con_trans_con_int['Modelo'], x_test_modeloBackwardAIC_con_trans_con_int, y_test))

# MODELO 3 Stepwise, métrica AIC  con transformaciones con interacciones

modeloStepAIC_con_trans_con_int = glm_stepwise(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'AIC')

# Resumen del modelo
print(summary_glm(modeloStepAIC_con_trans_con_int['Modelo'], y_train, modeloStepAIC_con_trans_con_int['X']))

# R-squared del modelo para train
print(pseudoR2(modeloStepAIC_con_trans_con_int['Modelo'], modeloStepAIC_con_trans_con_int['X'],  y_train))

# Preparo datos test
x_test_modeloStepAIC_con_trans_con_int = crear_data_modelo(x_test, modeloStepAIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloStepAIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloStepAIC_con_trans_con_int['Variables']['inter'])

# R-squared del modelo para test
print(pseudoR2(modeloStepAIC_con_trans_con_int['Modelo'], x_test_modeloStepAIC_con_trans_con_int, y_test))


# MODELO 4 forward, métrica BIC  con transformaciones con interacciones
modeloForwardBIC_con_trans_con_int = glm_forward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'BIC')

# Resumen del modelo
print(summary_glm(modeloForwardBIC_con_trans_con_int['Modelo'], y_train, modeloForwardBIC_con_trans_con_int['X']))

# R-squared del modelo para train
print(pseudoR2(modeloForwardBIC_con_trans_con_int['Modelo'], modeloForwardBIC_con_trans_con_int['X'], y_train))

# Preparo datos test
x_test_modeloForwardBIC_con_trans_con_int = crear_data_modelo(x_test, modeloForwardBIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloForwardBIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloForwardBIC_con_trans_con_int['Variables']['inter'])

# R-squared del modelo para test
print(pseudoR2(modeloForwardBIC_con_trans_con_int['Modelo'], x_test_modeloForwardBIC_con_trans_con_int, y_test))

# MODELO 5 backward, métrica BIC  con transformaciones con interacciones
modeloBackwardBIC_con_trans_con_int = glm_backward(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'BIC')

# Resumen del modelo
print(summary_glm(modeloBackwardBIC_con_trans_con_int['Modelo'], y_train, modeloBackwardBIC_con_trans_con_int['X']))

# R-squared del modelo para train
print(pseudoR2(modeloBackwardBIC_con_trans_con_int['Modelo'], modeloBackwardBIC_con_trans_con_int['X'], y_train))

# Preparo datos test
x_test_modeloBackwardBIC_con_trans_con_int = crear_data_modelo(x_test, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloBackwardBIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloBackwardBIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(pseudoR2(modeloBackwardBIC_con_trans_con_int['Modelo'], x_test_modeloBackwardBIC_con_trans_con_int, y_test))

# MODELO 6 Stepwise, métrica BIC  con transformaciones con interacciones
modeloStepBIC_con_trans_con_int = glm_stepwise(y_train, x_train, var_cont_con_transf, var_categ,
                                interacciones_unicas, 'BIC')

# Resumen del modelo
print(summary_glm(modeloStepBIC_con_trans_con_int['Modelo'], y_train, modeloStepBIC_con_trans_con_int['X']))

# R-squared del modelo para train
print(pseudoR2(modeloStepBIC_con_trans_con_int['Modelo'], modeloStepBIC_con_trans_con_int['X'], y_train))

# Preparo datos test
x_test_modeloStepBIC_con_trans_con_int = crear_data_modelo(x_test, modeloStepBIC_con_trans_con_int['Variables']['cont'], 
                                                    modeloStepBIC_con_trans_con_int['Variables']['categ'], 
                                                    modeloStepBIC_con_trans_con_int['Variables']['inter'])
# R-squared del modelo para test
print(pseudoR2(modeloStepBIC_con_trans_con_int['Modelo'], x_test_modeloStepBIC_con_trans_con_int, y_test))

#########################################################################################################################################
########################################### TABLA RESUMEN ###############################################################################
# Suponiendo que tienes estas listas con la información necesaria
modelos = ['Modelo aleatorio', 'Modelo Backward AIC', 'Modelo Forward AIC', 'Modelo Stepwise AIC', 'Modelo Backward BIC', 'Modelo Forward BIC', 'Modelo Stepwise BIC']

num_parametros = [
                    len(modelo_aleatorio['Modelo'].coef_[0]),
                    len(modeloBackwardAIC_con_trans_con_int['Modelo'].coef_[0]), 
                    len(modeloForwardAIC_con_trans_con_int['Modelo'].coef_[0]), 
                    len(modeloStepAIC_con_trans_con_int['Modelo'].coef_[0]), 
                    len(modeloBackwardBIC_con_trans_con_int['Modelo'].coef_[0]), 
                    len(modeloForwardBIC_con_trans_con_int['Modelo'].coef_[0]), 
                    len(modeloStepBIC_con_trans_con_int['Modelo'].coef_[0])
                ]

r2_test = [
            pseudoR2(modelo_aleatorio['Modelo'], x_test_aleatorio, y_test),
            pseudoR2(modeloBackwardAIC_con_trans_con_int['Modelo'], x_test_modeloBackwardAIC_con_trans_con_int, y_test), 
            pseudoR2(modeloForwardAIC_con_trans_con_int['Modelo'], x_test_modeloForwardAIC_con_trans_con_int, y_test),
            pseudoR2(modeloStepAIC_con_trans_con_int['Modelo'], x_test_modeloStepAIC_con_trans_con_int, y_test),
            pseudoR2(modeloBackwardBIC_con_trans_con_int['Modelo'], x_test_modeloBackwardBIC_con_trans_con_int, y_test), 
            pseudoR2(modeloForwardBIC_con_trans_con_int['Modelo'], x_test_modeloForwardBIC_con_trans_con_int, y_test),
            pseudoR2(modeloStepBIC_con_trans_con_int['Modelo'], x_test_modeloStepBIC_con_trans_con_int, y_test)
            ]

r2_train = [
            pseudoR2(modelo_aleatorio['Modelo'], modelo_aleatorio['X'], y_train),
            pseudoR2(modeloBackwardAIC_con_trans_con_int['Modelo'], modeloBackwardAIC_con_trans_con_int['X'], y_train),
            pseudoR2(modeloForwardAIC_con_trans_con_int['Modelo'], modeloForwardAIC_con_trans_con_int['X'], y_train),
            pseudoR2(modeloStepAIC_con_trans_con_int['Modelo'], modeloStepAIC_con_trans_con_int['X'], y_train),
            pseudoR2(modeloBackwardBIC_con_trans_con_int['Modelo'], modeloBackwardBIC_con_trans_con_int['X'], y_train),
            pseudoR2(modeloForwardBIC_con_trans_con_int['Modelo'], modeloForwardBIC_con_trans_con_int['X'], y_train),
            pseudoR2(modeloStepBIC_con_trans_con_int['Modelo'], modeloStepBIC_con_trans_con_int['X'], y_train)
            ]

# Crear el DataFrame
df = pd.DataFrame({
    'Modelo': modelos,
    'Número de Parámetros': num_parametros,
    'R2 Train': r2_train,
    'R2 Test': r2_test
})

# Mostrar el DataFrame
print(df)

df.to_csv('tabla_resultados_regresion_logistica.csv')

# Mejor modelo según el Área bajo la Curva ROC
AUC0 = curva_roc(x_test_aleatorio, y_test, modelo_aleatorio)
AUC1 = curva_roc(x_test_modeloBackwardAIC_con_trans_con_int, y_test, modeloBackwardAIC_con_trans_con_int)
AUC2 = curva_roc(x_test_modeloForwardAIC_con_trans_con_int, y_test, modeloForwardAIC_con_trans_con_int)
AUC3 = curva_roc(x_test_modeloStepAIC_con_trans_con_int, y_test, modeloStepAIC_con_trans_con_int)
AUC4 = curva_roc(x_test_modeloBackwardBIC_con_trans_con_int, y_test, modeloBackwardBIC_con_trans_con_int)
AUC5 = curva_roc(x_test_modeloForwardBIC_con_trans_con_int, y_test, modeloForwardBIC_con_trans_con_int)
AUC6 = curva_roc(x_test_modeloStepBIC_con_trans_con_int, y_test, modeloStepBIC_con_trans_con_int)

# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en seis modelos diferentes y almacena sus R-squared en listas separadas
    modelo_stepBIC = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloStepBIC_con_trans_con_int['Variables']['cont']
        , modeloStepBIC_con_trans_con_int['Variables']['categ']
        , modeloStepBIC_con_trans_con_int['Variables']['inter']
    )
    modelo_stepAIC = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloStepAIC_con_trans_con_int['Variables']['cont']
        , modeloStepAIC_con_trans_con_int['Variables']['categ']
        , modeloStepAIC_con_trans_con_int['Variables']['inter']
    )
    modelo_backBIC = validacion_cruzada_glm(
        5
        , x_train
        , y_train
        , modeloBackwardBIC_con_trans_con_int['Variables']['cont']
        , modeloBackwardBIC_con_trans_con_int['Variables']['categ']
        , modeloBackwardBIC_con_trans_con_int['Variables']['inter']
    )

    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'AUC': modelo_stepBIC + modelo_stepAIC + modelo_backBIC
        , 'Resample': ['Rep' + str((rep + 1))]*5*3  # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
    })
    results = pd.concat([results, results_rep], axis = 0)


# Boxplot de la validacion cruzada 
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráficoç
# Agrupa los valores de AUC por modelo
grupo_metrica = results.groupby('Modelo')['AUC']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('AUC')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico  

 
# Calcular la media del AUC por modelo
results.groupby('Modelo')['AUC'].mean()
# Calcular la desviación estándar del AUC por modelo
results.groupby('Modelo')['AUC'].std()    
# Guardamos resultados
results.to_csv("resultados_regresion_logistica.csv")
print(results)


# CALCULAMOS PUNTO DE CORTE
# Generamos una rejilla de puntos de corte
posiblesCortes = np.arange(0, 1.01, 0.01).tolist()  # Generamos puntos de corte de 0 a 1 con intervalo de 0.01
rejilla = pd.DataFrame({
    'PtoCorte': [],
    'Accuracy': [],
    'Sensitivity': [],
    'Specificity': [],
    'PosPredValue': [],
    'NegPredValue': []
})  # Creamos un DataFrame para almacenar las métricas para cada punto de corte

for pto_corte in posiblesCortes:  # Iteramos sobre los puntos de corte
    rejilla = pd.concat(
        [rejilla, sensEspCorte(modeloBackwardBIC_con_trans_con_int['Modelo'], x_test, y_test, pto_corte, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter'])],
        axis=0
    )  # Calculamos las métricas para el punto de corte actual y lo agregamos al DataFrame

rejilla['Youden'] = rejilla['Sensitivity'] + rejilla['Specificity'] - 1  # Calculamos el índice de Youden
rejilla.index = list(range(len(rejilla)))  # Reindexamos el DataFrame para que los índices sean consecutivos

plt.plot(rejilla['PtoCorte'], rejilla['Youden'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Youden')
plt.title('Youden')
plt.show()

plt.plot(rejilla['PtoCorte'], rejilla['Accuracy'])
plt.xlabel('Posibles Cortes')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()

print(rejilla['PtoCorte'][rejilla['Youden'].idxmax()])
print(rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()])

# El resultado es 0.41 para youden y 0.46 para Accuracy
# Los comparamos
sensEspCorte(modeloBackwardBIC_con_trans_con_int['Modelo'], x_test, y_test, 0.41, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter'])
sensEspCorte(modeloBackwardBIC_con_trans_con_int['Modelo'], x_test, y_test, 0.46, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter'])

# Vemos las variables mas importantes del modelo ganador
impVariablesLog(modeloBackwardBIC_con_trans_con_int, y_train, x_train, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter'])

# Vemos los coeficientes del modelo ganador
coeficientes = modeloBackwardBIC_con_trans_con_int['Modelo'].coef_
nombres_caracteristicas = crear_data_modelo(x_train, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter']).columns  # Suponiendo que X_train es un DataFrame de pandas
# Imprime los nombres de las características junto con sus coeficientes
for nombre, coef in zip(nombres_caracteristicas, coeficientes[0]):
    print(f"Variable: {nombre}, Coeficiente: {coef}")

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
pseudoR2(modeloBackwardBIC_con_trans_con_int['Modelo'], modeloBackwardBIC_con_trans_con_int['X'], y_train)
pseudoR2(modeloBackwardBIC_con_trans_con_int['Modelo'], x_test_modeloBackwardBIC_con_trans_con_int, y_test)
# Es poca la diferencia, por lo que el modelo se puede considerar robusto

# Calculamos la diferencia del Area bajo la curva ROC en train y test
curva_roc(crear_data_modelo(x_train, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter']), y_train, modeloBackwardBIC_con_trans_con_int)
curva_roc(x_test_modeloBackwardBIC_con_trans_con_int, y_test, modeloBackwardBIC_con_trans_con_int)

# Calculamos la diferencia de las medidas de calidad entre train y test 
sensEspCorte(modeloBackwardBIC_con_trans_con_int['Modelo'], x_train, y_train, 0.46, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter'])
sensEspCorte(modeloBackwardBIC_con_trans_con_int['Modelo'], x_test, y_test, 0.46, modeloBackwardBIC_con_trans_con_int['Variables']['cont'], modeloBackwardBIC_con_trans_con_int['Variables']['categ'], modeloBackwardBIC_con_trans_con_int['Variables']['inter'])

# Summary del modelo ganador
print(summary_glm(modeloBackwardBIC_con_trans_con_int['Modelo'], y_train, modeloBackwardBIC_con_trans_con_int['X']))