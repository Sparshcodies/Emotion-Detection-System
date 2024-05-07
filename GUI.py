import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load your CNN model
model = load_model("model.h5")  # Replace "model.h5" with your model file path


def get_prediction(face):
    face = cv2.resize(face, (48, 48))
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_gray = np.expand_dims(face_gray, axis=-1)
    prediction = model.predict(np.array([face_gray]))
    label = np.argmax(prediction)
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    return emotions[label]

# Function to get prediction from the CNN model for an image file
def get_prediction_from_file(file_path):
    frame = cv2.imread(file_path)
    if frame is not None:
        faces = faceCascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_face = frame[y:y + h, x:x + w]
            pred = get_prediction(cropped_face)
            cv2.putText(frame, str(pred), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame
    else:
        messagebox.showerror("Error", "Failed to load image.")
        return None

# Function to get prediction from the CNN model for webcam feed
def get_prediction_from_webcam():
    while True:
        ret, frame = video_capture.read()
        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            faces = faceCascade.detectMultiScale(
                cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cropped_face = frame[y:y + h, x:x + w]
                pred = get_prediction(cropped_face)
                cv2.putText(frame, str(pred), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            display_prediction(frame)
            window.update()  # Update the Tkinter window to show the new frame
        else:
            messagebox.showerror("Error", "Failed to capture video from camera.")
            break

# Function to display prediction result in the Tkinter window
def display_prediction(frame):
    if frame is not None:
        # Convert frame to ImageTk format and display in the GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        video_label.img = img
        video_label.config(image=img)
        video_label.image = img  # Keep reference to avoid garbage collection
    else:
        messagebox.showerror("Error", "Failed to display prediction.")

# Function to perform prediction based on user choice
def perform_prediction():
    if option.get() == "Image from Computer":
        # Release the video capture object if webcam feed is active
        if video_capture.isOpened():
            video_capture.release()
        file_path = filedialog.askopenfilename()
        if file_path:
            frame = get_prediction_from_file(file_path)
            display_prediction(frame)
    elif option.get() == "Webcam":
        frame = get_prediction_from_webcam()
        display_prediction(frame)
        

# Create the Tkinter window
window = tk.Tk()
window.title("Emotion Prediction")

# Set the minimum size of the window
window.minsize(400, 300)

# Function to handle window resize events
def on_resize(event):
    # Get the new size of the window
    new_width = event.width
    new_height = event.height

# Bind the resize event to the window
window.bind("<Configure>", on_resize)

# Create a label to prompt the user to select an option
option_label = tk.Label(window, text="Select an option:")
option_label.pack()

# Create a variable to store the selected option
option = tk.StringVar()
option.set(None)
# Create radio buttons for user selection
radio_button1 = tk.Radiobutton(window, text="Image from Computer", variable=option, value="Image from Computer")
radio_button1.pack(anchor=tk.W)
radio_button2 = tk.Radiobutton(window, text="Webcam", variable=option, value="Webcam")
radio_button2.pack(anchor=tk.W)

# Create a button to perform prediction based on user choice
predict_button = tk.Button(window, text="Predict", command=perform_prediction)
predict_button.pack()

# Create a label to display the prediction result
video_label = tk.Label(window)
video_label.pack()

# Load the Haar cascade classifier for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

# Run the Tkinter event loop
window.mainloop()

# Release the video capture object
video_capture.release()
