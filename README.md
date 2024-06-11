PRUEBA ACCURO - Reconocimiento facial

Incialmente plantee hacer todo en un mismo archivo, pero acabe haciendo un script separado para la generación de la bbdd de imágenes, este proceso se lleva a cabo partiendo de unos videos en formato mp4
Finalmente acabe añadiendo en este script el procesado de las imágenes (puntos 1 y 2 del enunciado).
La generación de los conjuntos de entrenamiento y prueba es manual, principalmente para limpiar el conjunto de entrenamiento de posibles falsos positivos; lo cual no interesa en el conjunto de prueba.
El conjunto de entrenamiento son las imagenes extraidas del video miguel1.mp4 y miguel3.mp4 habiendo eliminado los falsos positivos (capteras PruebaAccuro/bbdd/brutos/copia/miguel1 y miguel3, pasadas a PruebaAccuro/entrenamiento).
El conjunto de evaluación/prueba son los extraidos directamente de miguel2.mp4 como los devuelve el script procesadoGuardado.py

El proyecto esta pensado para poder rconocer a mas de una persona, solo haría falta añadir un conjunto de imagenes de entrenamiento de esa persona en PruebaAccuro/entrenamiento/[NombrePersona]

DEPENDENCIAS
    Python 3.9.2
    opencv-contrib-python 4.10.0.82
    numpy 1.24.2
    imutils 0.5.4