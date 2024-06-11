import cv2
import os
import imutils

# La función de este script es extraer imágenes de videos en .mp4, hacer el procesamiento indicado y guardar en .jpg 

video = 'miguel1'
dataPath = 'bbdd/brutos'
personPath = dataPath + '/' + video

if not os.path.exists(personPath):
    os.makedirs(personPath)
    print('Carpeta creada: ',personPath)

# Genera imagenes 
    # A partir de streaming
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    # A partir del video indicado
cap = cv2.VideoCapture("bbdd/brutos/"+video+".mp4")

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

while True:
    
    try:

        ret, frame = cap.read()
        # Si falla la lectura rompe el bucle
        if ret == False: break
        # Escalado solo en función de la anchura
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayEQ = cv2.equalizeHist(gray)

        faces = faceClassif.detectMultiScale(grayEQ,1.3,5)

        # Holgura del recorte respecto al rectangulo
        aux = 10

        # Encuadra todas las "caras" detectadas en la imagen
        for (x,y,w,h) in faces:
            # Pintado del rectangulo
            grayEQ = cv2.rectangle(grayEQ, (x,y),(x+w,y+h),(0,255,0),2)
            # Recorte con holgura del recuadro pintado
            rostro = grayEQ[y-aux:y+h+aux,x-aux:x+w+aux]
            # Estandariza el tamaño de las imagenes antes de guardarlas
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_'+video+'{}.jpg'.format(count),rostro)
            count = count + 1

        # Lineas comentadas para visualización del proceso 
        # cv2.imshow('frame',grayEQ)
        # k =  cv2.waitKey(1)
        # if k == 27 or count >= 400:
            
        # Hasta un maximo de 400 capturas
        if count >= 400:
            break

    except cv2.error as err:
            print("Error: {}".format(err))

cap.release()
# cv2.destroyAllWindows()