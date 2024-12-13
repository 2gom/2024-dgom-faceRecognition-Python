import cv2

imge_path = "images/imagen_009.jpg"
image = cv2.imread(imge_path)

#A escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Carga modelo de detección facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Detección de rostros
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) > 0:
    print(f"Se detectaron {len(faces)} rostros en la imagen")
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("No se detectaron rostros en la imagen")

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()