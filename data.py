import cv2 #libreria que permite analizar rostros
import os
import imutils

personName = 'Rostro Prueba' #nombre de la persona que será analizada
dataPath = 'Data/'  # Cambia a la ruta donde hayas almacenado Data
personPath = dataPath + '/' + personName #carpeta que se va a usar

if not os.path.exists(personPath): #si no exite la carpeta con el nombre de la persona con este codigo se crea
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #se utiliza la camara del dispositivo en el que se esta ejecutando
cap = cv2.VideoCapture('ejemplo7.mp4') #se utiliza un video que se tenga en el dispositivo

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0  # con el uso de hearcascade se puede analizar este archivo donde estan analizados los rasgos faciales

while True:
    #Mientras este en ejecucion esta parte del codigo va a captar los rostros, ya sea en uso de la camara
    #como tambien analizando un video
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #rectangulo que va a aparecer en
        #cuanto se encuentre un rostro
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count), rostro) #se va a guardar en formato jpg
        count = count + 1
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300: #el bucle termina cuando se guardan 300 datos
        break

cap.release()
cv2.destroyAllWindows()
