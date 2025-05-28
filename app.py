import os
import cv2
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Enable CORS for all origins
CORS(app, origins="*")  # You can specify more specific origins if needed, e.g., origins="http://127.0.0.1:5500"

# Initialize the OpenCV face detector
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract face embedding
def extract_face_embedding(frame, model_name="VGG-Face"):
    """Extract face embedding using DeepFace with a locally stored model."""
    try:
        temp_image = "temp_image.jpg"
        cv2.imwrite(temp_image, frame)

        # Load the model from your local directory
        embedding = DeepFace.represent(img_path=temp_image, model_name=model_name, enforce_detection=True)
        return np.array(embedding[0]["embedding"])
    except Exception as e:
        print(f"Face embedding extraction failed: {e}")
        return None

# Function to validate the live face
def validate_face(live_embedding, stored_embedding, threshold=0.363):
    """Validate the live face embedding against the stored embedding."""
    similarity = cosine_similarity([live_embedding], [stored_embedding])[0][0]
    return similarity > threshold

# Function to save the face embedding
def save_face_embedding(identifier, embedding, images):
    """Save the face embedding and the set of images for 3D reconstruction."""
    embedding_json = json.dumps(embedding.tolist())

    # Save the images as well for 3D model generation (you can use these later for training)
    for i, img in enumerate(images):
        cv2.imwrite(f"static/{identifier}_image_{i}.jpg", img)

    with open(f"static/{identifier}_embedding.json", "w") as file:
        file.write(embedding_json)

# Function to load the face embedding from file
def load_face_embedding(identifier):
    """Load the stored face embedding."""
    file_path = f"static/{identifier}_embedding.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as file:
        data = json.load(file)
        return np.array(data)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# SocketIO event to capture live video frames
@socketio.on('start_video')
def handle_video_stream(data):
    cap = cv2.VideoCapture(0)  # Start the webcam
    identifier = data.get("identifier")
    stored_embedding = load_face_embedding(identifier)

    if stored_embedding is None:
        emit('face_validation_error', {"message": f"No stored embedding found for identifier '{identifier}'."})
        return

    frame_rate = 5  # Process every 5th frame
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            emit('face_validation_error', {"message": "Failed to access the camera."})
            break

        frame_count += 1
        if frame_count % frame_rate == 0:
            # Resize for faster processing
            resized_frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = resized_frame[y:y + h, x:x + w]
                live_embedding = extract_face_embedding(face_roi)
                if live_embedding is not None:
                    if validate_face(live_embedding, stored_embedding):
                        emit('face_validated', {"message": "Face Validated!"})
                        cap.release()
                        return
                    else:
                        emit('face_validation_error', {"message": "Face validation failed. Trying again..."})
                        break

            # Send the live video frame to the frontend
            _, jpeg_frame = cv2.imencode('.jpg', resized_frame)
            frame_bytes = jpeg_frame.tobytes()
            emit('video_frame', {'frame': frame_bytes})

    cap.release()

# SocketIO event to register face (capture images and save embedding)
@socketio.on('register_face')
def register_face(data):
    identifier = data.get("identifier")
    embeddings = []
    images = []
    cap = cv2.VideoCapture(0)  # Start the webcam

    captured_count = 0
    frame_rate = 5  # Process every 5th frame
    frame_count = 0
    emit('status_update', {"message": "Please rotate your face to capture images from different angles."})

    while captured_count < 5:
        ret, frame = cap.read()
        if not ret:
            emit('status_update', {"message": "Failed to access the camera."})
            break

        frame_count += 1
        if frame_count % frame_rate == 0:
            # Resize for faster processing
            resized_frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = resized_frame[y:y + h, x:x + w]
                live_embedding = extract_face_embedding(face_roi)
                if live_embedding is not None:
                    embeddings.append(live_embedding)
                    images.append(resized_frame)
                    captured_count += 1
                    emit('status_update', {"message": f"Captured {captured_count}/5 images. Rotate your face for the next shot."})

            # Send the live video frame to the frontend
            _, jpeg_frame = cv2.imencode('.jpg', resized_frame)
            frame_bytes = jpeg_frame.tobytes()
            emit('video_frame', {'frame': frame_bytes})

    cap.release()

    if embeddings:
        # Save face embeddings and images
        save_face_embedding(identifier, embeddings[0], images)
        emit('status_update', {"message": f"Face for identifier '{identifier}' has been successfully registered!"})
        emit('model_saved', {"message": "3D model saved! You can now start validating the face."})
    else:
        emit('status_update', {"message": "Failed to extract face embedding."})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
