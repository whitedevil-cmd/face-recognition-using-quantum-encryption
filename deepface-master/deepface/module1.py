import sys
sys.path.append('/path/to/deepface-master')  # Update with the correct path to the deepface-master folder

from deepface import DeepFace

def extract_face_embedding(frame, model_name="Facenet"):
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