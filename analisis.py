# Entrenamos con los n-1 tramo, y dejamos un tramo entero para testear
# Descartamos años de mejora
# python file.py main --p ---> hace el main y grafica

# Descartamos errores y mejoras
# y luego buscamos el polinomio que mejor se aproxime

# El polinomio lo buscamos creando (anioMaximo-anioMinimo)*10
   # luego buscamos por "aproximacion" el valor de polinomio(anio) para sacar ruido


import os
import sys
import argparse
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from scipy import interpolate
from sklearn import preprocessing

def descarta_tramo_mejoras(datos, lastTramo):
    """
    Filtra tramos donde el IRI mejora más que la tolerancia del error de medicion.
    Args:
        datos: datos a analizar

    Returns: datos filtrados

    """
    etiquetas_tramos = np.unique(datos[:, 0])
    bad_indexes = []
    bad_indexes_ERROR = []
    
    for lbl in etiquetas_tramos:
        current_lbl_indexes = datos[:, 0] == lbl
        
        # Nos devuelve en True aquellas filas del tramo actual del "for"

        # Buscamos las filas donde aparece el 1ero. y ultimo registro de cada tramo
        # la vamos a usar para eliminar registros si hubo mejoras
        indice_minimo_tramo = np.min(np.argwhere(datos[:, 0] == lbl))
        indice_maximo_tramo = np.max(np.argwhere(datos[:, 0] == lbl))
        #print(indice_minimo_tramo)
        #print(indice_maximo_tramo)
        #print(datos)

        # diff calcula la diferencia sobre el mismo eje
        # en este caso, sobre la columna 6 del tramo entero
        iri_diff = np.diff(datos[current_lbl_indexes, 6])

        any_high_values_ERROR = np.any((iri_diff <= -0.3) & (iri_diff > -0.6))
        # En "any_high_values" nos devuelve True/False de aquellos tramos que verifica la condicion anterior

        if any_high_values_ERROR:
            max_high_value_ERROR = np.max(np.argwhere((iri_diff <= -0.3) & (iri_diff > -0.6)) + 1)
            # max_high_value nos devuelve la maxima fila (por tramo)
            # donde verifica la diferencia

            datos[indice_minimo_tramo + max_high_value_ERROR, 6] = datos[indice_minimo_tramo + max_high_value_ERROR-1, 6]
            # En "bad_indexes_ERROR" tenemos las filas que se van a eliminar porque hubo ERROR de medicion:
                # en lugar de descartarla, copiamos el valor del iri anterior
                # porque sino, cuando hago la polinomica, tengo que "inventar" los otros valores del vector que acabamos de eliminar

        # DESCARTAMOS MEJORAS
        # En lugar de descartarla vamos a colocarlas como nuevo tramo (desde año 0)
        any_high_values = np.any(iri_diff <= -0.6)
        if any_high_values:
            max_high_value = np.max(np.argwhere(iri_diff <= -0.6) + 1)
            anioMejora = datos[indice_minimo_tramo + max_high_value, 1]
            anioMax = datos[indice_maximo_tramo, 1]
            anio = 0
            if anioMejora <= anioMax / 2:
                fila = indice_minimo_tramo + max_high_value
                while fila <= indice_maximo_tramo:
                    datos[fila, 1] = anio  #Corregimos año
                    fila += 1
                    anio += 1

                fila = indice_minimo_tramo
                while fila < indice_minimo_tramo + max_high_value:
                    datos[fila, 0] = lastTramo + 1  #Corregimos tramo
                    fila += 1
                    
            else:
                fila = indice_minimo_tramo
                while fila < indice_minimo_tramo + max_high_value:
                    datos[fila, 0] = lastTramo + 1  #Corregimos tramo
                    fila += 1
                
                fila = indice_minimo_tramo + max_high_value
                while fila <= indice_maximo_tramo:
                    datos[fila, 1] = anio  #Corregimos año
                    fila += 1
                    anio += 1
                      
            lastTramo += 1

    return datos


def search_polilinea(datos):

    etiquetas_tramos = np.unique(datos[:, 0])
    
    for lbl in etiquetas_tramos:
       current_lbl_indexes = datos[:, 0] == lbl

       indice_minimo_tramo = np.min(np.argwhere(current_lbl_indexes))
       indice_maximo_tramo = np.max(np.argwhere(current_lbl_indexes))
       largo_tramo = indice_maximo_tramo - indice_minimo_tramo + 1
       anioMin = datos[indice_minimo_tramo, 1]
       anioMax = datos[indice_maximo_tramo, 1]

       #Hacemos la polilinea si tenemos mas de dos puntos
       if anioMax > 1:  
           iri = [None] * largo_tramo
           
           # En "x" se guarda los años absolutos que quedaron sin los errores de medicion y sin las mejoras...
           # En "y" se guarda los iris correspondientes al "x" anterior...
           x = datos[indice_minimo_tramo:indice_maximo_tramo + 1, 1]
           y = datos[indice_minimo_tramo:indice_maximo_tramo + 1, 6]
           
           # poly crea la tupla (x,y) anterior
           poly = [tuple((a, b)) for a, b in zip(datos[indice_minimo_tramo:indice_maximo_tramo + 1, 1],
                                                 datos[indice_minimo_tramo:indice_maximo_tramo + 1, 6])]
           
           l = len(x)
           # linspace(start,stop,num,endpoint=True,retstep=False)
           t = np.linspace(0, 1, l-2, endpoint=False)   # Crea un array con valor inicial "start", valor final "stop" y "num" elementos
           t = np.append([0, 0, 0], t)
           t = np.append(t, [1, 1, 1])

           tck0 = [t, [x,y], 3]
           elem = (anioMax-anioMin)*10
           u3 = np.linspace(0, 1, elem, endpoint=False)
           out0 = interpolate.splev(u3, tck0)
           outArray = np.array(out0)

           # En outArray[0,i] figuran todos los puntos del eje x
           # En outArray[1,i] figuran todos los puntos del eje y
           index = indice_minimo_tramo
           num = 0
           while (index <= indice_maximo_tramo) and (num < int(elem)):
              # Recorremos outArray[0,num], entramos cuando encontramos el proximo mayor
              if outArray[0,num] > anioMin:
                 if anioMin-outArray[0,num-1] > outArray[0,num]-anioMin:
                    datos[index, 6] = outArray[1,num]
                 else:
                    datos[index, 6] = outArray[1,num-1]
                 anioMin += 1
                 index += 1
              num += 1
          
    return datos

def forzar_ascendente(datos, lastTramo):

    etiquetas_tramos = np.unique(datos[:, 0])

    # Vamos a "forzar" para que queden ascendente los iri
    test_data = list()
    train_lbl_draw = list()
    
    for lbl in etiquetas_tramos:
       current_lbl_indexes = datos[:, 0] == lbl

       indice_minimo_tramo = np.min(np.argwhere(current_lbl_indexes))
       indice_maximo_tramo = np.max(np.argwhere(current_lbl_indexes))

       iri_diff = np.diff(datos[current_lbl_indexes, 6])
       any_high_values_ASC = np.any(iri_diff < 0)

       while any_high_values_ASC:
          high_value_ASC = (iri_diff < 0).nonzero()[0] + 1
          min_high_value = np.min(np.argwhere(iri_diff < 0))

          iriMin = datos[indice_minimo_tramo + min_high_value, 6]
          datos[indice_minimo_tramo + min_high_value + 1, 6] = iriMin

          #Volvemos a fijarnos si existe algun valor en negativo
          iri_diff = np.diff(datos[current_lbl_indexes, 6])
          any_high_values_ASC = np.any(iri_diff < 0)


       # En indice_maximo_tramo se guarda la ultima medicion de cada tramo
       # ... donde cada tmda = tmdaAnterior * 1.02
       # ... deflex = deflexAnterior

       ind = indice_maximo_tramo
       tramo = datos[ind, 0]
       anio = datos[ind, 1]
       deflex = datos[ind, 2]
       tP = datos[ind, 3]*1.02
       tM = datos[ind, 4]*1.02
       tL = datos[ind, 5]*1.02
       iri = datos[ind, 6]
       #test_data.append([tramo, anio+anioPred, int(deflex), int(tP), int(tM), int(tL)])
       test_data.append([anio+1, int(deflex), int(tP), int(tM), int(tL), iri])

       train_lbl_draw.append([datos[ind, 6]])
       
    return datos, test_data, train_lbl_draw

def main(argv):
    """
    TransformApp entry point
    argv: command-line arguments (excluding this script's name)
    """

    '''
    El módulo "argparse" facilita la escritura de interfaces de línea de comandos fáciles de usar.
    El programa define qué argumentos requiere, y "argparse" descubrirá cómo analizar los que están fuera de sys.argv.
    El módulo "argparse" también genera automáticamente mensajes de ayuda y uso y emite errores cuando los usuarios dan al programa argumentos inválidos.
    '''
    parser = argparse.ArgumentParser(description="Script de modelado de evolucion de IRI")

    # Named arguments
    parser.add_argument("--verbose", "-v", help="Genera una salida detallada en consola", action="store_true")
    parser.add_argument("datos", help="Ruta al archivo de datos csv", metavar="DATA_PATH", nargs="?")
    parser.add_argument("--plot", "-p", help="Grafica las predicciones", action="store_true")

    args = parser.parse_args(argv)

    if not args.datos:
        print("Faltan argumentos posicionales")
        parser.print_usage()
        return 1

    if os.path.isfile(args.datos):
        data = np.loadtxt(args.datos, delimiter=",")
    else:
        # Cargo datos
        data = np.loadtxt('datos.csv', delimiter=",")

    # Nos quedamos con el nro. de tramo
    etiquetas_tramos=np.unique(data[:, 0])
    lastTramo = np.max(etiquetas_tramos)
    
    # Reemplazamos el año absoluto, por el relativo
    # Buscamos el menor año de evaluacion, restandoselo al año
    data[:, 1] -= np.min(data[:, 1])
    
    ####################################
    
    # descarto tramo cuando hay mejoras    
    data = descarta_tramo_mejoras(data, lastTramo)

     # En data tenemos los datos sin los errores de medicion, y sin las mejoras
    ####################################
    
    data = search_polilinea(data)
    # Ahora en data tenemos en la columna del IRI los iri corregidos

    data, test_data, train_lbl_draw = forzar_ascendente(data, lastTramo)
    # En data nos quedo el iri ascendente
    # En test_data estan los valores para testear 
    # En train_lbl_draw nos quedo los valores del iri de entrenamiento para poder "plotear"    

    # Pero antes guardamos la rugosidad (la usamos para predecir)
    data_lbl = data[:,-1].copy()
    data[:, -1] = np.roll(data[:, -1], 1)
   
    for tramoTrain in np.unique(data[:, 0]):
        current_lbl_indexes = data[:, 0] == tramoTrain
        indice_minimo_tramo = np.min(np.argwhere(current_lbl_indexes))
        indice_maximo_tramo = np.max(np.argwhere(current_lbl_indexes))
        anioMax = data[indice_maximo_tramo, 1]

        #Hacemos la polilinea si tenemos mas de dos puntos
        if anioMax > 1:           
            # itero por los casos de corte para arreglar el iri
            case = indice_minimo_tramo
            m = data[case + 2, -1] - data[case + 1, -1]
            h = data[case + 1, -1]
            # evaluo en la recta en el x anterior (-1)
            jj = -1 * m + h - 0.5
            data[case, -1] = jj   # metemos ruido gaussiano sigma 0.5

    
    ################## ENTRENAMOS ##################
    train_data = data[:, 1:]
    test_data = np.array(test_data)

    # Guardamos la rugosidad (la usamos para predecir)
    train_data_lbl = data_lbl

    anioPred = 1
    n = test_data.shape
    size_test = n[0]
    
    while anioPred <= 3:
        #clf = SVR(C=1, kernel='rbf', epsilon=0.005)
        #clf.fit(train_data, train_data_lbl)
        
        rfc = RFR(n_estimators=5, random_state=1)
        rfc.fit(train_data, train_data_lbl)
        
        #yPredict_SVR = clf.predict(test_data)
        yPredict_RFR = rfc.predict(test_data)
        
        i = 0
        while i < size_test:
            test_data[i, 0] = test_data[i, 0] + 1   #anio
            test_data[i, 1] = test_data[i, 1]       #deflexión
            test_data[i, 2] = test_data[i, 2]*1.02  #transito pesado
            test_data[i, 3] = test_data[i, 3]*1.02  #transito pesado
            test_data[i, 4] = test_data[i, 4]*1.02  #transito pesado
            test_data[i, 5] = yPredict_RFR[i]
            #test_data[i, 5] = yPredict_SVR[i]

            i += 1
            
        anioPred += 1
    
    if args.plot:
        #Graficamos
        plt.plot(train_lbl_draw, c='k', marker="o", fillstyle='full', label='valor real del ultimo año')
        plt.xlabel("Tramo")	#Insertamos el titulo del eje X
        plt.ylabel("Rugosidad") #Insertamos el titulo del eje Y

        #plt.plot(yPredict_SVR, c='c', marker="o", fillstyle='full', label="SVR a 3 años futuro")
        plt.plot(yPredict_RFR, c='r', marker="o", fillstyle='full', label="RFR a 3 años futuro")

        plt.grid()
        plt.title("RF")
        plt.legend(loc=0)
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
