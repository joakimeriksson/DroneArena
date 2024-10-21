# Copyright (c) 2024 RISE
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Face recognition using DeepFace - for the drone arena "demo"
# Author: Joakim Eriksson, joakim.eriksson@ri.se

import cv2
import os
import numpy as np
import json
from deepface import DeepFace

# Function to load known faces
def load_known_faces(directory):
    known_faces = {}
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(directory, filename)
            known_faces[name] = DeepFace.represent(img_path, model_name="VGG-Face")[0]["embedding"]
    return known_faces


def face_recognition(frame, known_faces):
    # Detect faces
    try:
        faces = DeepFace.extract_faces(frame, enforce_detection=False)
    except Exception as e:
        print(f"Error in face detection: {e}")
        faces = []

    # Process each detected face
    for face in faces:
        if face['confidence'] > 0.8:
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Extract face region
            face_image = frame[y:y+h, x:x+w]
            
            # Get face embedding
            try:
                embedding = DeepFace.represent(face_image, model_name="VGG-Face")[0]["embedding"]
                
                # Compare with known faces
                best_match = None
                best_distance = float('inf')
                for name, known_embedding in known_faces.items():
                    distance = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
                    if distance < best_distance:
                        best_distance = distance
                        best_match = name

                # Draw rectangle and label
                color = (200, 255, 200) if best_distance < 0.6 else (200, 200, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{best_match}: {best_distance:.2f}" if best_match else "Unknown"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)
                
            except Exception as e:
                print(f"Error in face comparison: {e}")
    return faces

if __name__ == "__main__":
    # Open the default camera (0). If you have multiple cameras, you can use 1, 2, etc.
    camera = cv2.VideoCapture(0)

        # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Load known faces
    known_faces_dir = "photos"
    known_faces = load_known_faces(known_faces_dir)

    cap = camera
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Detect faces - will also "mark" the faces in the frame
        faces = face_recognition(frame, known_faces)
        if len(faces) > 0:
            try:
                predictions = DeepFace.analyze(frame)
                print(json.dumps(predictions, indent=4))
            except Exception as e:
                print(f"Error in face analysis: {e}")
 
        # Display the resulting frame
        cv2.imshow('Real-time Face Detection and Recognition', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

