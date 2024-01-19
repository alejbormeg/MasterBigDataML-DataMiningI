# Cargo las librerias 
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Cargo las funciones que voy a utilizar
from FuncionesMineria import (analizar_variables_categoricas, cuentaDistintos, frec_variables_num, 
                           atipicosAmissing, patron_perdidos, ImputacionCuant, ImputacionCuali, lm_custom)

import random

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
from FuncionesMineria import (graficoVcramer, mosaico_targetbinaria, boxplot_targetbinaria, 
                           hist_targetbinaria, Transf_Auto, lm, Rsq, validacion_cruzada_lm,
                           modelEffectSizes, modelEffectSizes_custom, crear_data_modelo, Vcramer)

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

# Obtengo la importancia de las variables
graficoVcramer(datos_input, varObjBin)
graficoVcramer(datos_input, varObjCont)

# Eliminamos Codigo Provincia y Superficie pues por el significado de las variables no guardan relación con lo que queremos predecir
# Además su importancia para la variable objetivo es muy pequeña
columns_to_remove = ["CodigoProvincia", "SUPERFICIE"]
datos_input = datos_input.drop(columns = columns_to_remove)

# Crear un DataFrame para almacenar los resultados del coeficiente V de Cramer
VCramer = pd.DataFrame(columns=['Variable', 'Objetivo', 'Vcramer'])

# Genera una lista con los nombres de las variables.
variables = list(datos_input.columns)

for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjCont)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjCont.name, 'Vcramer': v_cramer},
                             ignore_index=True)
    
for variable in variables:
    v_cramer = Vcramer(datos_input[variable], varObjBin)
    VCramer = VCramer.append({'Variable': variable, 'Objetivo': varObjBin.name, 'Vcramer': v_cramer},
                             ignore_index=True)

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
plt.figure(figsize=(8, 6))
# Establecer el tamaño de fuente en el gráfico
sns.set(font_scale=1.2)
# Crear un mapa de calor (heatmap) de la matriz de correlación
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, mask=mask)
# Establecer el título del gráfico
plt.title("Matriz de correlación")
# Mostrar el gráfico de la matriz de correlación
plt.show()

# Busco las mejores transformaciones para las variables numericas con respesto a los dos tipos de variables
input_cont = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjCont)], axis = 1)
input_bin = pd.concat([datos_input, Transf_Auto(datos_input[numericas], varObjBin)], axis = 1)

# Creamos conjuntos de datos que contengan las variables explicativas y una de las variables objetivo y los guardamos
todo_cont = pd.concat([input_cont, varObjCont], axis = 1)
todo_bin = pd.concat([input_bin, varObjBin], axis = 1)
with open('todo_bin.pickle', 'wb') as archivo:
    pickle.dump(todo_bin, archivo)
with open('todo_cont.pickle', 'wb') as archivo:
    pickle.dump(todo_cont, archivo)


## Comenzamos con la regresion lineal

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
# Separate numeric and categorical variables
var_cont1 = []
var_categ1 = []

for column in input_cont.columns:
    if pd.api.types.is_numeric_dtype(input_cont[column]):
        var_cont1.append(column)
    elif pd.api.types.is_categorical_dtype(input_cont[column]) or pd.api.types.is_string_dtype(input_cont[column]):
        if(column != 'prop_missings'):
            var_categ1.append(column)

# Obtengo la particion a partir de los datos depurados
datos_modelo = crear_data_modelo(input_cont, var_cont1, var_categ1)

var_cont1 = []
var_categ1 = []

for column in datos_modelo.columns:
    if pd.api.types.is_numeric_dtype(datos_modelo[column]):
        var_cont1.append(column)
    elif pd.api.types.is_categorical_dtype(datos_modelo[column]) or pd.api.types.is_string_dtype(datos_modelo[column]):
        if(column != 'prop_missings'):
            var_categ1.append(column)

x_train, x_test, y_train, y_test = train_test_split(datos_modelo, np.ravel(varObjCont), test_size = 0.2, random_state = 123456)

# Creo el modelo
# Prepara los datos para el modelo, incluyendo la codificación de variables categóricas y la creación de interacciones.

modelo1 = lm_custom(y_train, x_train, var_cont1, var_categ1)

# Visualizamos los resultado del modelo
print(modelo1['Modelo'].summary())

# Calculamos la medida de ajuste R^2 para los datos de entrenamiento
print(Rsq(modelo1['Modelo'], y_train, modelo1['X']))

# # Preparamos los datos test para usar en el modelo
# x_test_modelo1 = crear_data_modelo(x_test, var_cont1, var_categ1)
# Calculamos la medida de ajuste R^2 para los datos test
print(Rsq(modelo1['Modelo'], y_test, x_test))


# Nos fijamos en la importancia de las variables
print(modelEffectSizes_custom(modelo1, y_train, x_train, var_cont1, var_categ1))


# Vamos a probar un modelo con menos variables. Recuerdo el grafico de Cramer
graficoVcramer(datos_input, varObjCont) # Pruebo con las mas importantes

# Construyo el segundo modelo con las 5 con mayor V cramer + Densidad + Actividad ppal
# Columns to include
included_columns = ["ComercTTEHosteleria", "Servicios", "Construccion", "Industria"]
var_cont2 = x_train.filter(regex='^(CCAA_|Densidad_|ActividadPpal_)|' + '|'.join(included_columns)).columns.tolist()
var_categ2 = []
modelo2 = lm(y_train, x_train, var_cont2, var_categ2)
print(modelo2['Modelo'].summary())
print(Rsq(modelo2['Modelo'], y_train, modelo2['X']))
x_test_modelo2 = crear_data_modelo(x_test, var_cont2, var_categ2)
print(Rsq(modelo2['Modelo'], y_test, x_test_modelo2))
print(modelEffectSizes_custom(modelo2, y_train, x_train, var_cont2, var_categ2))


# Pruebo un modelo con menos variables, basandome en la importancia de las variables
# Number of columns to randomly select
num_columns_to_select = 3

# Randomly select column names
random_columns = x_train.columns.tolist()
selected_columns = random.sample(random_columns, k=num_columns_to_select)
var_cont3 = selected_columns
var_categ3 = []
modelo3 = lm(y_train, x_train, var_cont3, var_categ3)
print(modelo3['Modelo'].summary())
print(Rsq(modelo3['Modelo'], y_train, modelo3['X']))
x_test_modelo3 = crear_data_modelo(x_test, var_cont3, var_categ3)
print(Rsq(modelo3['Modelo'], y_test, x_test_modelo3))
print(modelEffectSizes_custom(modelo3, y_train, x_train, var_cont3, var_categ3))

# # Pruebo con una interaccion sobre el anterior
# # Se podrian probar todas las interacciones dos a dos
# var_cont4 = []
# var_categ4 = ['Etiqueta', 'CalifProductor', 'Clasificacion']
# var_interac4 = [('Clasificacion', 'Etiqueta')]
# modelo4 = lm(y_train, x_train, var_cont4, var_categ4, var_interac4)
# modelo4['Modelo'].summary()
# Rsq(modelo4['Modelo'], y_train, modelo4['X'])
# x_test_modelo4 = crear_data_modelo(x_test, var_cont4, var_categ4, var_interac4)
# Rsq(modelo4['Modelo'], y_test, x_test_modelo4)


# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'Rsquared': [],
    'Resample': [],
    'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_lm(5, x_train, y_train, var_cont1, var_categ1)
    modelo2VC = validacion_cruzada_lm(5, x_train, y_train, var_cont2, var_categ2)
    modelo3VC = validacion_cruzada_lm(5, x_train, y_train, var_cont3, var_categ3)
    # modelo4VC = validacion_cruzada_lm(5, x_train, y_train, var_cont4, var_categ4, var_interac4)
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'Rsquared': modelo1VC + modelo2VC + modelo3VC,
        'Resample': ['Rep' + str((rep + 1))] * 5 * 3,  # Etiqueta de repetición
        'Modelo': [1] * 5 + [2] * 5 + [3] * 5 # Etiqueta de modelo (1, 2, 3 o 4)
    })
    
    # Concatena los resultados de esta repetición al DataFrame principal 'results'
    results = pd.concat([results, results_rep], axis=0)

print(results)

# Boxplot de la validación cruzada
plt.figure(figsize=(10, 6))  # Crea una figura de tamaño 10x6
plt.grid(True)  # Activa la cuadrícula en el gráfico
# Agrupa los valores de R-squared por modelo
grupo_metrica = results.groupby('Modelo')['Rsquared']
# Organiza los valores de R-squared por grupo en una lista
boxplot_data = [grupo_metrica.get_group(grupo).tolist() for grupo in grupo_metrica.groups]
# Crea un boxplot con los datos organizados
plt.boxplot(boxplot_data, labels=grupo_metrica.groups.keys())  # Etiqueta los grupos en el boxplot
# Etiqueta los ejes del gráfico
plt.xlabel('Modelo')  # Etiqueta del eje x
plt.ylabel('Rsquared')  # Etiqueta del eje y
plt.show()  # Muestra el gráfico 
    

# Calcular la media de las métricas R-squared por modelo
media_r2 = results.groupby('Modelo')['Rsquared'].mean()
print(media_r2)
# Calcular la desviación estándar de las métricas R-squared por modelo
std_r2 = results.groupby('Modelo')['Rsquared'].std()
print(std_r2)
# Contar el número de parámetros en cada modelo
num_params = [len(modelo1['Modelo'].params), len(modelo2['Modelo'].params)]

# Teniendo en cuenta el R2, la estabilidad y el numero de parametros, nos quedamos con el modelo3
# Vemos los coeficientes del modelo ganador
modelo2['Modelo'].summary()

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
Rsq(modelo2['Modelo'], y_train, modelo2['X'])
Rsq(modelo2['Modelo'], y_test, x_test_modelo2)

# Vemos las variables mas importantes del modelo ganador
modelEffectSizes(modelo2, y_train, x_train, var_cont2, var_categ2)


# Regresión Logística

# Cargo las librerias 

# Cargo las funciones que voy a utilizar despues
from FuncionesMineria import (graficoVcramer, impVariablesLog, pseudoR2, glm, summary_glm, 
                           validacion_cruzada_glm, sensEspCorte, crear_data_modelo, curva_roc, glm_custom)

# Pruebo un primer modelo con las variables originales
datos_logistic = datos_modelo.copy()

# Obtengo la particion
x_train, x_test, y_train, y_test = train_test_split(datos_logistic, varObjBin, test_size = 0.2, random_state = 1234567)
# Indico que la variable respuesta es numérica (hay que introducirla en el algoritmo de phython tal y como la va a tratar)
y_train, y_test = y_train.astype(int), y_test.astype(int)

# Construyo un modelo preliminar con todas las variables (originales)
# Indico la tipología de las variables (numéricas o categóricas)
var_cont1 = []
var_categ1 = []

for column in datos_logistic.columns:
    if pd.api.types.is_numeric_dtype(datos_logistic[column]):
        var_cont1.append(column)
    elif pd.api.types.is_categorical_dtype(datos_logistic[column]) or pd.api.types.is_string_dtype(datos_logistic[column]):
        if(column != 'prop_missings'):
            var_categ1.append(column)

# Creo el modelo inicial
modeloInicial = glm_custom(y_train, x_train, var_cont1, var_categ1)

# Visualizamos los resultado del modelo
print(summary_glm(modeloInicial['Modelo'], y_train, modeloInicial['X']))

# Calculamos la medida de ajuste R^2 para los datos de entrenamiento
print(pseudoR2(modeloInicial['Modelo'], modeloInicial['X'], y_train))

# Preparamos los datos test para usar en el modelo
x_test_modeloInicial = crear_data_modelo(x_test, var_cont1, var_categ1)

# Calculamos la medida de ajuste R^2 para los datos test
print(pseudoR2(modeloInicial['Modelo'], x_test_modeloInicial, y_test))

# Calculamos el número de parámetros utilizados en el modelo.
print(len(modeloInicial['Modelo'].coef_[0]))

# Fijandome en la significacion de las variables, el modelo con las variables mas significativas queda
included_columns = ['Age_over65_pct', 'WomanPopulationPtge', 'SameComAutonPtge', 'DifComAutonPtge', 'ConstructionUnemploymentPtge', 'totalEmpresas', 'Industria', 'Construccion', 'ComercTTEHosteleria', 'Servicios', 'Pob2010', 'SUPERFICIE', 'PobChange_pct', 'PersonasInmueble']
var_cont2 = x_train.filter(regex='^(CCAA_|Densidad_|ActividadPpal_)|' + '|'.join(included_columns)).columns.tolist()
var_categ2 = []
modelo2 = glm(y_train, x_train, var_cont2, var_categ2)

print(summary_glm(modelo2['Modelo'], y_train, modelo2['X']))

print(pseudoR2(modelo2['Modelo'], modelo2['X'], y_train))

x_test_modelo2 = crear_data_modelo(x_test, var_cont2, var_categ2)
print(pseudoR2(modelo2['Modelo'], x_test_modelo2, y_test))

print(len(modelo2['Modelo'].coef_[0]))

# Calculamos y representamos la importancia de las variables en el modelo
impVariablesLog(modelo2, y_train, x_train, var_cont2, var_categ2)

# Calculamos el area bajo la curva ROC y representamos
AUC2 = curva_roc(x_test_modelo2, y_test, modelo2)

# Miro el grafico V de Cramer para ver las variables mas importantes
graficoVcramer(datos_input, varObjBin)

included_columns = ['totalEmpresas', 'Pob2010', 'PersonasInmueble', 'Inmuebles', 'Age_under19_Ptge', 'Population', 'Age_0-4_Ptge', 'TotalCensus', 'Servicios', 'Construccion', 'ComercTTEHosteleria', 'Age_over65_pct', 'Industria']
var_cont3 = x_train.filter(regex='^(CCAA_|Densidad_|ActividadPpal_)|' + '|'.join(included_columns)).columns.tolist()
var_categ3 = []

modelo3 = glm(y_train, x_train, var_cont3, var_categ3)

print(summary_glm(modelo3['Modelo'], y_train, modelo3['X']))

print(pseudoR2(modelo3['Modelo'], modelo3['X'], y_train))

x_test_modelo3 = crear_data_modelo(x_test, var_cont3, var_categ3)
print(pseudoR2(modelo3['Modelo'], x_test_modelo3, y_test))

print(len(modelo3['Modelo'].coef_[0]))


# Pruebo alguna interaccion sobre el modelo 3
var_cont4 = var_cont3
var_categ4 = var_categ3
var_interac4 = [('Clasificacion', 'CalifProductor')]
modelo4 = glm(y_train, x_train, var_cont4, var_categ4, var_interac4)
summary_glm(modelo4['Modelo'], y_train, modelo4['X'])
pseudoR2(modelo4['Modelo'], modelo4['X'], y_train)
x_test_modelo4 = crear_data_modelo(x_test, var_cont4, var_categ4, var_interac4)
pseudoR2(modelo4['Modelo'], x_test_modelo4, y_test)
len(modelo4['Modelo'].coef_[0])



# Pruebo uno con las variables mas importantes del 2 
var_cont5 = []
var_categ5 = ['Clasificacion', 'CalifProductor', 'Etiqueta']
modelo5 = glm(y_train, x_train, var_cont5, var_categ5)
summary_glm(modelo5['Modelo'], y_train, modelo5['X'])
pseudoR2(modelo5['Modelo'], modelo5['X'], y_train)
x_test_modelo5 = crear_data_modelo(x_test, var_cont5, var_categ5)
pseudoR2(modelo5['Modelo'], x_test_modelo5, y_test)
len(modelo5['Modelo'].coef_[0])

# Pruebo uno con las variables mas importantes del 2 y una interaccion
var_cont6 = []
var_categ6 = ['Clasificacion', 'CalifProductor', 'Etiqueta']
var_interac6 = [('Clasificacion', 'Etiqueta')]
modelo6 = glm(y_train, x_train, var_cont6, var_categ6, var_interac6)
summary_glm(modelo6['Modelo'], y_train, modelo6['X'])
pseudoR2(modelo6['Modelo'], modelo6['X'], y_train)
x_test_modelo6 = crear_data_modelo(x_test, var_cont6, var_categ6, var_interac6)
pseudoR2(modelo6['Modelo'], x_test_modelo6, y_test)
len(modelo6['Modelo'].coef_[0])

# Mejor modelo según el Área bajo la Curva ROC
AUC1 = curva_roc(x_test_modeloInicial, y_test, modeloInicial)
AUC2 = curva_roc(x_test_modelo2, y_test, modelo2)
AUC3 = curva_roc(x_test_modelo3, y_test, modelo3)
AUC4 = curva_roc(x_test_modelo4, y_test, modelo4)
AUC5 = curva_roc(x_test_modelo5, y_test, modelo5)
AUC6 = curva_roc(x_test_modelo6, y_test, modelo6)

# Hago validacion cruzada repetida para ver que modelo es mejor
# Crea un DataFrame vacío para almacenar resultados
results = pd.DataFrame({
    'AUC': []
    , 'Resample': []
    , 'Modelo': []
})

# Realiza el siguiente proceso 20 veces (representado por el bucle `for rep in range(20)`)
for rep in range(20):
    # Realiza validación cruzada en cuatro modelos diferentes y almacena sus R-squared en listas separadas
    modelo1VC = validacion_cruzada_glm(5, x_train, y_train, var_cont1, var_categ1)
    modelo2VC = validacion_cruzada_glm(5, x_train, y_train, var_cont2, var_categ2)
    modelo3VC = validacion_cruzada_glm(5, x_train, y_train, var_cont3, var_categ3)
    modelo4VC = validacion_cruzada_glm(5, x_train, y_train, var_cont4, var_categ4, var_interac4)
    modelo5VC = validacion_cruzada_glm(5, x_train, y_train, var_cont5, var_categ5)
    modelo6VC = validacion_cruzada_glm(5, x_train, y_train, var_cont6, var_categ6, var_interac6)
    
    # Crea un DataFrame con los resultados de validación cruzada para esta repetición
    results_rep = pd.DataFrame({
        'AUC': modelo1VC + modelo2VC + modelo3VC + modelo4VC + modelo5VC + modelo6VC
        , 'Resample': ['Rep' + str((rep + 1))]*5*6  # Etiqueta de repetición (5 repeticiones 6 modelos)
        , 'Modelo': [1]*5 + [2]*5 + [3]*5 + [4]*5 + [5]*5 + [6]*5 # Etiqueta de modelo (6 modelos 5 repeticiones)
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
# Contar el número de parámetros en cada modelo
num_params = [len(modeloInicial['Modelo'].coef_[0]), len(modelo2['Modelo'].coef_[0]), len(modelo3['Modelo'].coef_[0]), 
 len(modelo4['Modelo'].coef_[0]), len(modelo5['Modelo'].coef_[0]), len(modelo6['Modelo'].coef_[0])]

print(num_params)

## Buscamos el mejor punto de corte

# Probamos dos
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.4, var_cont5, var_categ5)
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.6, var_cont5, var_categ5)

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
        [rejilla, sensEspCorte(modelo5['Modelo'], x_test, y_test, pto_corte, var_cont5, var_categ5)],
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

rejilla['PtoCorte'][rejilla['Youden'].idxmax()]
rejilla['PtoCorte'][rejilla['Accuracy'].idxmax()]

# El resultado es 0.75 para youden y 0.5 para Accuracy
# Los comparamos
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.75, var_cont5, var_categ5)
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.5, var_cont5, var_categ5)

# Vemos las variables mas importantes del modelo ganador
impVariablesLog(modelo5, y_train, x_train, var_cont5, var_categ5)

# Vemos los coeficientes del modelo ganador
coeficientes = modelo5['Modelo'].coef_
nombres_caracteristicas = crear_data_modelo(x_train, var_cont5, var_categ5).columns  # Suponiendo que X_train es un DataFrame de pandas
# Imprime los nombres de las características junto con sus coeficientes
for nombre, coef in zip(nombres_caracteristicas, coeficientes[0]):
    print(f"Variable: {nombre}, Coeficiente: {coef}")

# Evaluamos la estabilidad del modelo a partir de las diferencias en train y test:
pseudoR2(modelo5['Modelo'], modelo5['X'], y_train)
pseudoR2(modelo5['Modelo'], x_test_modelo5, y_test)
# Es poca la diferencia, por lo que el modelo se puede considerar robusto

# Calculamos la diferencia del Area bajo la curva ROC en train y test
curva_roc(crear_data_modelo(x_train, var_cont5, var_categ5), y_train, modelo5)
curva_roc(x_test_modelo5, y_test, modelo5)

# Calculamos la diferencia de las medidas de calidad entre train y test 
sensEspCorte(modelo5['Modelo'], x_train, y_train, 0.5, var_cont5, var_categ5)
sensEspCorte(modelo5['Modelo'], x_test, y_test, 0.5, var_cont5, var_categ5)





#####################################################################
########## Correlation matrix code ##################################
# Miramos la matriz de correlación entre las variables
# Crea un mapa de calor
correlation_matrix = datos_input.corr()
# Obtén una máscara para la parte triangular inferior
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Configura el tamaño de la figura
plt.figure(figsize=(10, 8))

# Crea un mapa de calor utilizando la máscara
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)

plt.title('Matriz de Correlación - Triangular Inferior')
plt.show()

# Eliminamos aquellas variables que tienen linearidad perfecta o alta (la mayoría información redundante)
# columns_to_remove = ["TotalCensus", "Pob2010", "inmuebles", "Servicios", "ComercTTEHosteleria", "Construccion", "Industria", "totalEmpresas",
#                      "PersonasInmueble", "Age_0-4_Ptge", "SameComAutonPtge"]

# columns_to_remove = ["TotalCensus", "Pob2010", "inmuebles", "Servicios", "ComercTTEHosteleria", "Construccion", "Industria", "totalEmpresas"]

# datos_input = datos_input.drop(columns=columns_to_remove)

# Miramos la matriz de correlación entre las variables
# Crea un mapa de calor
correlation_matrix = datos_input.corr()
# Obtén una máscara para la parte triangular inferior
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Configura el tamaño de la figura
plt.figure(figsize=(10, 8))

# Crea un mapa de calor utilizando la máscara
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, mask=mask)

plt.title('Matriz de Correlación - Triangular Inferior')
plt.show()



#####################################################################
##################### Recategorizar #################################

# Recategorizo uniendo Ceuta, Melilla y Murcia por tener una actividad principal similar (Hostelería, servicios u otros)
# y por tener esto influencia notable sobre la variable objetivo, como vemos más adelante
datos['CCAA'] = datos['CCAA'].replace({'Ceuta': 'Murcia_Ceuta_Melilla','Melilla': 'Murcia_Ceuta_Melilla', 'Murcia': 'Murcia_Ceuta_Melilla'})

# Uno industria y construcción porque las ciudades en las que predominan suelen tener la misma tendencia electoral
datos['ActividadPpal'] = datos['ActividadPpal'].replace({'Construccion': 'Industria_Construccion','Industria': 'Industria_Construccion'})

# Frecuencias de los valores en las variables categóricas
analisis_categoricas = analizar_variables_categoricas(datos)

print(analisis_categoricas)