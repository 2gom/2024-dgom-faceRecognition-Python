from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Cargar el modelo de detección de caras de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route("/detect_faces", methods=["POST"])
def detect_faces():
    # Obtener la imagen de la solicitud
    if 'image' not in request.files:
        return jsonify({"error": "No se proporcionó una imagen"}), 400

    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)  #Lee la imagen como arreglo
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Preparar la respuesta
    face_list = [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x, y, w, h) in faces]
    return jsonify({"num_faces": len(faces), "faces": face_list})

if __name__ == "__main__":
    app.run(debug=True)