# Driver Drowsiness Alert System

## Project Description
The Driver Drowsiness Alert System is designed to monitor a driver's eye movements and detect signs of drowsiness. It uses computer vision and machine learning techniques to help alert the driver when prolonged eye closure is detected.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy

## Project Files
- `driver_drowsiness.py` – Main project code
- `train_model.py` – Model training code
- `drowsiness_model.pkl` – Trained machine learning model
- `face_landmarker.task` – Face landmark model

## How It Works
The system detects facial landmarks and calculates the Eye Aspect Ratio (EAR). If the eyes remain closed beyond a set duration, an alert is triggered.

## Project Type
Machine Learning and Computer Vision
