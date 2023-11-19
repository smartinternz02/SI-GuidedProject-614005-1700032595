import streamlit as st
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
import pickle
import time

# Create a dictionary to store the state
state = {'run': False, 'end_button_clicked': False}

# Load the pre-trained model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def run_detection():
    # Display the webcam feed and body language detection
    camera = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    end_button_placeholder = st.empty()
    end_button_displayed = False

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera.isOpened():
            ret, frame = camera.read()

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image[:, :, 1] = image[:, :, 1] * 1.0  # Increase saturation
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            # Adjust brightness and contrast
            alpha = 1.5  # Contrast control (1.0 means no change)
            beta = 25  # Brightness control (0 means no change)

            adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

            # Make Detections
            results = holistic.process(image)

            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

            # Recolor image back to BGR for rendering
            image.flags.writeable = True

            # 1. Draw face landmarks
            # 1. Draw face landmarks without grid
            # 1. Draw face landmarks without dots
            if results.face_landmarks is not None:
                pass  # Do not draw anything

            # 2. Right hand without dots
            if results.right_hand_landmarks is not None:
                pass  # Do not draw anything

            # 3. Left hand without dots
            if results.left_hand_landmarks is not None:
                pass  # Do not draw anything

            # 4. Pose Detections without dots
            if results.pose_landmarks is not None:
                pass  # Do not draw anything

            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in
                                          pose]).flatten())

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in
                                          face]).flatten())

                # Concatenate rows
                row = pose_row + face_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)

                # Grab ear coords
                coords = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                    , [640, 480]).astype(int))

                cv2.rectangle(image,
                              (coords[0], coords[1] + 5),
                              (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                              (255, 255, 255),  # White box
                              -1)

                cv2.putText(image, body_language_class, coords,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Get status box
                cv2.rectangle(image, (0, 0), (250, 60), (255, 255, 255), -1)

                # Display Class
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                            2, cv2.LINE_AA)

                # Display Probability
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            except:
                pass

            FRAME_WINDOW.image(image, channels="RGB", output_format="JPEG")

            if state['run'] and not state['end_button_clicked'] and not end_button_displayed:
                end_button_clicked = end_button_placeholder.button("End", key="end_button")
                state['end_button_clicked'] = end_button_clicked
                end_button_displayed = True

            if state['end_button_clicked']:
                break

    camera.release()
    cv2.destroyAllWindows()

def homepage():
    st.image("1.jpg", width=700)
    st.markdown("<hr style='border: 5px solid #000000; width: 100%;'>", unsafe_allow_html=True)
    st.title("📘Know About Emotion Detection ")
    st.image("2.png",width=900)
    st.markdown("<hr style='border: 5px solid #000000; width: 100%;'>", unsafe_allow_html=True)
    st.title("🌐 Identifying Body Language ")
    st.write("--> Just click on 'Run' button to let us explore your moods and emotions.")
   

    st.markdown("<hr style='border: 5px solid #000000; width: 100%;'>", unsafe_allow_html=True)
    st.title("📚Instructions")
    st.write("-> Click on the Run button and wait for 20 seconds for execution of the program.")
    st.write("-> Do not move frequently as it may confuse the AI trained model.")
    st.write("-> Look at all the gestures to test all the emotions and to get the required result.")
    # Add more content as needed
    if st.button("Run"):
        st.write("Running the program... Please wait.")
        state['run'] = True
        run_detection()
        st.success("Program executed successfully!")

def about():
    st.title("📘 About Us")
    st.write("Discover the story behind body language detection. Your journey into AI begins here!")
    # Add more content as needed

def instructors():
    st.title("👩‍🏫 Instructors")
    st.write("Meet our passionate and experienced instructors. Learn from the best in the industry.")
    st.write("-> Vasundhara")
    st.write("-> Riya")
    st.write("-> Aditya Bajaj")
    # Add more content as needed

def contact():
    st.title("📧 Contact Us")
    st.write("Have questions or feedback? Contact us at vasundhara25003@gmail.com")
    st.write("OR")
    # Add more content as needed
    st.write("Vasundhara - https://www.linkedin.com/in/vasundhara-vashishtha-0ba939231/")
    st.write("Riya - https://www.linkedin.com/in/riya-arora-212739231/")
    st.write("Aditya Bajaj - https://www.linkedin.com/in/aditya-bajaj-18a14327a")

def user_feedback():
    st.title("✨ User Feedback")
    feedback = st.text_area("Share your feedback:")
    if st.button("Submit Feedback"):
        st.success("Feedback submitted successfully! Thank you for your input.")
        # Add logic to handle the feedback (store it, send notifications, etc.)

def main():
    # Add a header space with enhanced text style, "Hey Hi" text, logo on the left, body emoticon on the left, and centered text on the right
    st.markdown(
        """
        <div style='display: flex; justify-content: center; align-items: center; padding: 1em;'>
            <div>
                <p style='color: #ffffff; font-size: 1.2em; font-family: "Roboto", sans-serif;'> </p>
            </div>
            <div>
                <h1 style='color: #001F3F; font-size: 2.5em; font-family: "Roboto", sans-serif;'>BODY✨LANGUAGE✨DETECTION</h1>
                <hr style='border: 5px solid #000000; width: 100%;'>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Add logo in the navigation panel at the top
    logo_image = st.sidebar.image("3.jpg", width=100)

# Set a margin to push the logo to the top
    st.sidebar.markdown("<style>div.Widget.row-widget.stRadio > div{flex-direction: column;}</style>", unsafe_allow_html=True)
    st.sidebar.title("🚀 Navigation")
    pages = ["Home", "About", "Instructors", "Contact", "Feedback"]
    selection = st.sidebar.radio("Go to", pages)

    st.markdown(
        """
        <style>
            body {
                background-color: #001f3f !important;  /* Dark Blue */
                color: #001f3f;
                font-family: "Roboto", sans-serif;
            }
            .sidebar .css-1d01bu2 {
                background-color: #343a40;
                padding: 1em;
                border-radius: 10px;
                color: #ffffff;
                text-align: center;  /* Center align text in the navigation bar */
            }
            .sidebar .css-17vfa1t {
                font-size: 2.5em;
                font-family: "Montserrat", sans-serif;
                margin-top: 1em;
                padding: 0.5em 1em;
                border-radius: 5px;
                text-align: center;  /* Center align text in the navigation bar */
            }
            .sidebar .css-17vfa1t:hover {
                background-color: #007bff;
                color: #ffffff;
                transition: background-color 0.3s, color 0.3s;
            }
            .widget-title {
                font-size: 1.5em;
                font-family: "Open Sans", sans-serif;
                margin-top: 1.5em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if selection == "Home":
        homepage()
    elif selection == "About":
        about()
    elif selection == "Instructors":
        instructors()
    elif selection == "Contact":
        contact()
    elif selection == "Feedback":
        user_feedback()

# Run the Streamlit app
if __name__ == "__main__":
    main()