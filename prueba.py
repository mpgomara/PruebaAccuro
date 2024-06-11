import cv2
import os
import numpy as np

def main():
#    ProcesadoGuardado()
   EntrenamientoModeloLBPH()
   TestModeloLBPH()


def ProcesadoGuardado():
    pahtEntrada = "bbdd/brutos/entrada"
    contenido = os.listdir(pahtEntrada)
    faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    #Bucle procesado y guardado de imagenes
    for filename in contenido:
        try:
        # Preprocesado
            inImage = cv2.imread(pahtEntrada + "/" + filename, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(inImage, cv2.COLOR_BGR2GRAY)
            grayEQImage = cv2.equalizeHist(gray)

        # Detección de rostros
            faces = faceClassifier.detectMultiScale(grayEQImage, 1.3, 5)

            # cv2.imshow("grayEQ",grayEQImage)
            # cv2.waitKey()
            print(grayEQImage.shape)
            print(inImage.shape)
            print(faces)

            if faces!=():
                # Encuadra todas las "caras" detectadas en la imagen
                for (x,y,w,h) in faces:
                    outImage = cv2.rectangle(grayEQImage,(x,y),(x+w,y+h),(0,255,0),2)
                    outImage = cv2.resize(outImage,(150,150), interpolation=cv2.INTER_CUBIC)
            else:
                outImage = cv2.resize(grayEQImage,(150,150), interpolation=cv2.INTER_CUBIC)
        
            # guardar imagen
            cv2.imwrite("bbdd/procesados/"+filename, outImage)
                # print(filename)

        except cv2.error as err:
            print("Error: {}".format(err))

def EntrenamientoModeloLBPH():
    if os.path.exists('modeloLBPHFace.xml'):
        print("Usando modelo ya almacenado")
        return
    
    grupo = os.listdir("entrenamiento")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    labels = []
    facesData = []
    label = 0

    #Bucle personas a reconocer
    for persona in grupo:
        # personPath = dataPath + '/' + persona
        print('Leyendo las imágenes')

        #bucle imagenes entrenamiento cada persona
        for fileName in os.listdir('entrenamiento/' + persona):
            print('Rostros: ', persona + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread('entrenamiento/' +persona+'/'+fileName, cv2.IMREAD_GRAYSCALE))
        label = label + 1

    # Entrenando el reconocedor de rostros
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('modeloLBPHFace.xml')
    print("Modelo almacenado...")

def TestModeloLBPH():
    grupo = os.listdir("prueba/miguel2")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Leyendo modelo entrenado
    face_recognizer.read("modeloLBPHFace.xml")

    # datos para evaluación
    caras = os.listdir("bbdd/brutos/copia/miguel2")

    stats = []
    n = len(grupo)
    nP = len(caras)
    detectadas = 0
    nFP = 0


    # Análisis y visualización clasificados como negativos
    # ClassN = []
    # ClassFN = []

    for aux in grupo:
        imagen = cv2.imread("prueba/miguel2/"+aux, cv2.IMREAD_GRAYSCALE)
        result = face_recognizer.predict(imagen)
        # Humbral de detección
        if result[1] < 70:
            if aux[0] == 'N':
                nFP = nFP + 1
            stats.append(result[1])
            detectadas = detectadas + 1

        # Análisis y visualización clasificados como negativos
        # else:
        #     # print(str(aux[0])+", "+str(result[1]))
        #     if aux[0] == 'N':
        #         ClassN.append(result[1])
        #         print(str(aux[0])+", "+str(result[1]))
        #         cv2.imshow("VN",imagen)
        #     else:
        #         ClassFN.append(result[1])
        #         # print(str(aux[0])+", "+str(result[1]))
        #         cv2.imshow("FN",imagen)
        #     cv2.waitKey(1)

    # print("Media FN: "+ str(np.mean(ClassFN)))
    # print("Max FN: "+ str(np.max(ClassFN)))
    # print("Min FN: "+ str(np.min(ClassFN)))

    nN = n - nP
    nFN = nP - detectadas
    nVP = detectadas - nFP
    nVN = nN - nFP

    print("\n##### EVALUACIÓN #####")
    print("Totales: " + str(n))
    print("Positivos: " + str(nP))
    print("Negativos: " + str(nN))
    print("--------------------")
    print("Verdaderos positivos: " + str(nVP))
    print("Falsos positivos: " + str(nFP))
    print("Verdaderos negativos: " + str(nVN))
    print("Falsos negativos: " + str(nFN))
    print("Sensibilidad: " + str(nVP/(nP+nFN)))
    print("Especificidad: " + str(nVN/(nFP+nVN)))
    print("Precisión: " + str(nVP/(nVP+nFP)))

    # cv2.destroyAllWindows()

    
if __name__ =='__main__':
    main()
  
