import threading
import queue
import cv2
import os
import imutils
import numpy as np

def inputTask(cola, comando_cola, dataPath):
    personPath = dataPath
    while True:
        comando = comando_cola.get()
        if comando == 'input':
            while os.path.exists(personPath):
                nombre = input("\nIntroduzca un identificativo: ")
                personPath = dataPath + '/' + nombre
            cola.put(nombre)
        elif comando == 'exit':
            break

def trainTask(cola_train, comando_cola_train, faceRecognizer):
    while True:
        comando = comando_cola_train.get()
        if comando == 'train':
            EntrenamientoModeloLBPH(faceRecognizer)
            cola_train.put("trained_true")
            # Hasta aqui ok
        elif comando == 'leer_modelo':
            print("Leyendo modelo...")
            faceRecognizer.read("modeloLBPHFace.xml")
        elif comando == 'exit':
            break

def EntrenamientoModeloLBPH(faceRecognizer):
    peopleList = os.listdir("entrenamiento")
    print("Lista de personas: " + str(peopleList))    
    
    labels = []
    facesData = []
    label = 0
 
    #Bucle personas a reconocer
    for person in peopleList:
        print('Leyendo las imágenes de '+ person)

        #bucle imagenes entrenamiento cada persona
        for fileName in os.listdir('entrenamiento/' + person):
            # print('Rostros: ' + person + '/' + fileName)
            labels.append(label)
            auxImg = cv2.imread('entrenamiento/' + person +'/'+ fileName, cv2.IMREAD_GRAYSCALE)
            if auxImg is None: 
                print("Error leyendo imagen")
            else:
                facesData.append(auxImg)

        label = label + 1

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    faceRecognizer.train(facesData, np.array(labels))
    faceRecognizer.write('modeloLBPHFace.xml')
    print("Modelo almacenado")



faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

dataPath = 'entrenamiento'
cola = queue.Queue()
comando_cola = queue.Queue()
inputThread = threading.Thread(target=inputTask, args=(cola, comando_cola, dataPath))
inputThread.start()

cola_train = queue.Queue()
comando_cola_train = queue.Queue()
trainThread = threading.Thread(target=trainTask, args=(cola_train, comando_cola_train, faceRecognizer))
trainThread.start()


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
k = -1
training = False
trained = False
count = 0
aux = 10
peopleList = os.listdir("entrenamiento")

if os.listdir("entrenamiento") == []:
    print("Aun no hay perfiles para detectar")

if os.path.exists('modeloLBPHFace.xml'):
    print("Usando modelo ya almacenado")
    print("Leyendo modelo...")
    faceRecognizer.read("modeloLBPHFace.xml")
    trained = True

print("\nPresione [Ctrl + <] para añadir un nuevo rostro a reconocer")


while k != 27:
    k = cv2.pollKey()

    ret, frame = cap.read()
    if not ret: break

    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayEQ = cv2.equalizeHist(gray)
    faces = faceClassifier.detectMultiScale(grayEQ,1.3,5)

    
    if not cola_train.empty():
        aux_comando_train = cola_train.get()
        if aux_comando_train =='trained_true':
            trained = True
            cola_train.put("leer_modelo")
            # faceRecognizer.read("modeloLBPHFace.xml")
            

    for (x,y,w,h) in faces:
        if training:
            if count==10:
                print("Extrayendo imagenes para entrenamiento...")
            cv2.rectangle(grayEQ, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = grayEQ[y-aux:y+h+aux,x-aux:x+w+aux]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_'+nombre+'_{}.jpg'.format(count),rostro)
            count = count + 1
            if count>400:
                training = False
                count = 0
                comando_cola_train.put('train')

            # B G R
            cv2.putText(frame,'Capturando',(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,128,255),2)
        else:
            if trained:
                cv2.rectangle(grayEQ, (x,y),(x+w,y+h),(0,255,0),2)
                rostro = grayEQ[y-aux:y+h+aux,x-aux:x+w+aux]
                rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
                result = faceRecognizer.predict(rostro)

                if result[1] < 70:
                    cv2.putText(frame,'{}'.format(peopleList[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            else:
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
            # B G R
            # cv2.putText(frame,'{}'.format(k),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            # cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)


    if k == 28:
        comando_cola.put('input')

    if not cola.empty():
        nombre = cola.get()
        personPath = dataPath + '/' + nombre
        cola.task_done()  # Llama a task_done después de obtener un elemento
        os.makedirs(personPath)
        print('Carpeta creada: ',personPath)
        peopleList = os.listdir("entrenamiento")
        training = True

    cv2.imshow('Reconocedor facial', frame)

# Señala al hilo de entrada que debe detenerse
comando_cola.put('exit')
comando_cola_train.put('exit')

# Espera a que el hilo de entrada termine
inputThread.join()
# trainThread.join()

# Libera la captura de video y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()