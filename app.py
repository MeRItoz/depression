import cv2
import numpy as np
from keras.models import model_from_json
import os
from flask import Flask, jsonify, Response
from flask_cors import CORS
from collections import Counter
import face_recognition
from mtcnn.mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

cap = None  # Initialize cap to None


def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)  # Open the default camera (usually the webcam)
    if not cap.isOpened():  # Check if the camera capture was successfully opened
        cap = None  # Set cap to None if opening the camera fails


initialize_camera()  # Initialize the camera when your application starts

detect_emotion = False
emotion_labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
mtcnn_detector = MTCNN()

# Load the emotion detection model and dictionary
script_dir = os.path.dirname(os.path.abspath(__file__))
model_json_path = os.path.join(script_dir, 'model', 'emotion_model.json')
model_weights_path = os.path.join(script_dir, 'model', 'emotion_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

json_file = open(model_json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(model_weights_path)
print("Loaded emotion model from disk")

known_face_encodings = []
known_face_names = []

# Modify the path to your dataset accordingly
dataset_path = os.path.join(script_dir, 'dataset')

# Create a dictionary to store user details
user_details = {}

# Initialize the camera capture when your application starts
cap = cv2.VideoCapture(0)  # Open the default camera (usually the webcam)
if not cap.isOpened():  # Check if the camera capture was successfully opened
    cap = None  # Set cap to None if opening the camera fails

# Read user details from a text file based on the name prefix of image files
with open('user_details.txt', 'r') as file:
    for line in file:
        values = line.strip().split(',')
        if len(values) >= 4:
            name, full_name, address, email = values[:4]
            user_details[name] = {
                "Email": email,
                "Address": address,
                "Name": full_name,

                # Add other details as needed
            }

for file_name in os.listdir(dataset_path):
    if file_name.endswith('.jpg'):
        name_prefix = os.path.splitext(file_name)[0].split('_')[
            0]  # Extract the main part of the name from the image file
        image_path = os.path.join(dataset_path, file_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            original_image = cv2.imread(image_path)
            rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
            face_encodings = face_recognition.face_encodings(rotated_image)

        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(name_prefix)  # Use the main part of the name as the name
        else:
            print(f"No face found in image: {image_path}")

# Confidence threshold for face recognition
FACE_RECOGNITION_THRESHOLD = 0.6  # Adjust this value as needed


def generate_frames():
    global cap  # Make sure to use the global cap variable

    if cap is None or not cap.isOpened():
        return  # Return if the camera is not properly initialized or opened

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break  # Break the loop if there's no valid frame

        _, buffer = cv2.imencode('.jpg', frame)
        if buffer is not None:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break  # Break the loop if encoding fails


@app.route('/single_image')
def single_image():
    global cap

    if cap is not None:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                raise Exception("Failed to capture a valid frame from the camera.")

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        except Exception as e:
            print("Error:", e)
            if cap is not None:
                cap.release()
            initialize_camera()  # Reinitialize the camera
            return jsonify({'error': 'Could not read a frame from the camera.'})

        return Response((b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return jsonify({'error': 'Camera not initialized'})


@app.route('/detect_emotion', methods=['GET', 'POST'])
def detect_emotion_endpoint():
    global detect_emotion, dominant_emotion

    detected_emotions = []
    detected_names = set()
    name_detected = False  # Initialize a flag for face name detection

    while detect_emotion:
        if cap is not None:  # Ensure the camera capture is initialized
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))

            # Detect faces using MTCNN
            faces = mtcnn_detector.detect_faces(frame)

            for face in faces:
                x, y, width, height = face['box']
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0),
                              2)  # Draw a green box around the face
                roi_frame = frame[y:y + height, x:x + width]

                # Preprocess the face for emotion detection
                resized_frame = cv2.resize(roi_frame, (48, 48))
                grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                input_data = np.expand_dims(np.expand_dims(grayscale_frame, -1), 0)

                emotion_prediction = emotion_model.predict(input_data)
                max_index = int(np.argmax(emotion_prediction))

                detected_emotions.append(emotion_labels[max_index])

                # Face recognition
                face_encoding = face_recognition.face_encodings(frame, [(y, x + width, y + height, x)])

                if face_encoding:
                    name_detected = True  # Set the name_detected flag to True

                    # Compare known face encodings with the current face
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0],
                                                             tolerance=FACE_RECOGNITION_THRESHOLD)
                    if any(matches):
                        name = known_face_names[matches.index(True)]
                        detected_names.add(name)
    
                cv2.imshow("Emotion Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if name_detected:
                break  # Break the loop when a name is detected
        else:
            break  # Break the loop when the camera is not initialized

    cap.release()  # Release the camera capture when done
    cv2.destroyAllWindows()

    print("Detected Emotions:", detected_emotions)
    print("Detected Names:", detected_names)  # Add this line to print detected names

    if detected_emotions:
        dominant_emotion = max(Counter(detected_emotions).items(), key=lambda x: x[1])[0]
        print("Dominant Emotion:", dominant_emotion)

    # Retrieve user details associated with the detected face
    if detected_names:
        user_info = user_details.get(list(detected_names)[0], {})
        print("User Info:", user_info)  # Add this line to print user info
    else:
        user_info = {}

    return jsonify(
        {'dominant_emotion': dominant_emotion, 'detected_names': list(detected_names), 'user_details': user_info})


@app.route('/start_emotion_detection', methods=['POST'])
def start_emotion_detection():
    global detect_emotion
    detect_emotion = True
    return jsonify({'message': 'Emotion detection started'})


if __name__ == '__main__':
    try:
        app.run()
    finally:
        if cap is not None:
            cap.release()

