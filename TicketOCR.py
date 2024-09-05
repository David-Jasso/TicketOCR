import cv2 as cv
import numpy as np
import re

global dst_img
global img 

src_points = []
dst_points = [(0,0), (350,0), (350,500), (0,500)]
 
dst_img = np.zeros((500, 350), dtype = np.uint8)

# Lista para almacenar letras, numeros y caracteres (vocabulario).
vocabulary =[]

# Abrir archivo para importar el vocabulario
with open("resources/alphabet_94.txt") as f:
    # Leer cada linea del archivo y agregarlo a la lista.
    for l in f:
        vocabulary.append(l.strip())
    f.close()
print("Vocabulary:", vocabulary)
print("Vocabulary size: ", len(vocabulary))

# Importar modelos para reconocimiento de texto.
# DB modelo de deteccion de texto basado en resnet50.
textDetector = cv.dnn_TextDetectionModel_DB("resources/DB_TD500_resnet18.onnx")

inputSize = (640, 640)

# Umbral para mapas binarios y poligonos.
binThresh = 0.3
polyThresh = 0.5

mean = (122.67891434, 116.66876762, 104.00698793)

textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
textDetector.setInputParams(1.0/255, inputSize, mean, True)

# Modelo CRNN para reconocimiento de texto.
textRecognizer = cv.dnn_TextRecognitionModel("./resources/crnn_cs.onnx")
textRecognizer.setDecodeType("CTC-greedy")
textRecognizer.setVocabulary(vocabulary)
textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)

# Funcion para extrar u transformar 
def fourPointsTransform(frame, vertices):
    """Extrae y transforma roi del cuadro definido por los vértices en un rectángulo."""
    # Obtiene los vértices de cada cuadro delimitador 
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array([
        [0, outputSize[1] - 1],
        [0, 0],
        [outputSize[0] - 1, 0],
        [outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")
    # Aplicar transformación de perpectiva.
    rotationMatrix = cv.getPerspectiveTransform(vertices, targetVertices)
    result = cv.warpPerspective(frame, rotationMatrix, outputSize)
    return result

# Funcion para realizar la Homografia a partir de cuatro puntos dados.
def onMouseEvent(event, x, y, flags, userData):
    #print(event, x, y)
    global src_points
    global img
    global dst_points
    global dst_img
    if event == cv.EVENT_LBUTTONUP:
        src_points.append((x,y))
        cv.circle(img, (x,y), 10, (255, 0, 0), -1)
        cv.imshow("Original", img)
        print(event, x, y)
 
        if len(src_points) == 4:
            h, status = cv.findHomography(np.array(src_points), np.array(dst_points))
            dst_img = cv.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
            
            # Usar el Detector para detectar la presencia de texto en la imagen.
            boxes, confs = textDetector.detect(dst_img)

            # Dibujar las cajas delimitadoras en la imagen.
            cv.polylines(dst_img, boxes, True, (255, 0, 255), 1)
            print(confs)

            text = []

            # Mostrar la salida transformada de la primera caja de texto detectada.
            for box in boxes:
                warped_detection = fourPointsTransform(dst_img, box)

                # Reconocer el texto utilizando el modelo crnn.
                recResult = textRecognizer.recognize(warped_detection)

                # Almacenar textos detectados.
                text.append(recResult)
            text = ' '.join(text)
            print(text)
        
            cv.imshow("Resultado", dst_img)
            #Comentar y descomentar según sea el caso.
            #Escribir Resultado del Ticket1.
            cv.imwrite('TicketOCR/Resultado1.jpg', dst_img)
            #Escribir Resultado del Ticket2.
            #cv.imwrite('TicketOCR/Resultado2.jpg', dst_img)
            #Escribir Resultado del Ticket3.
            #cv.imwrite('TicketOCR/Resultado3.jpg', dst_img)
            #Escribir Resultado del Ticket4.
            #cv.imwrite('TicketOCR/Resultado4.jpg', dst_img)
            
            #Comentar y descomentar según sea el caso.
            #Si es el ticket 1:
            match = re.search(r'17.34', text, re.IGNORECASE)
            #Si es el ticket 2:
            #match = re.search(r'598', text, re.IGNORECASE)
            #Si es el ticket 3:
            #match = re.search(r'11.27', text, re.IGNORECASE)
            #Si es el ticket 4:
            #match = re.search(r'980', text, re.IGNORECASE)
            if match:
                total = match.group()
                print(f"El total de la compra es: {total}")
            else:
                print("No se pudo encontrar el total en el ticket.")

# Cargar imagen para realizar homografia y reconocimiento de texto.
# Cambiar el tipo de imagen segun sea el caso: jpg, jpeg, png, etc.
img = cv.imread('TicketOCR/Ticket1.jpeg', cv.IMREAD_COLOR)
# Ajustar tamaño de imagen segun sea el caso.
img = cv.resize(img, None, fx = 0.5, fy = 0.5)

# Llamar a eventos del Mouse para obtener la imagen a la que se le realizara el reconocimiento de texto.
cv.namedWindow("Original")
cv.setMouseCallback("Original", onMouseEvent)
 
cv.imshow("Original", img)
cv.waitKey(0)
 
cv.destroyAllWindows()