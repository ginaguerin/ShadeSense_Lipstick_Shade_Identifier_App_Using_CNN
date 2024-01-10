import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import time
import datetime

# Set background image
background_image = "https://github.com/ginaguerin/ShadeSense_Lipstick_Shade_Identifier_App/blob/master/Application/background.jpeg?raw=true"

# Set the background style using st.markdown
background_style = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Optima&display=swap');

        .main {{
            background: url("{background_image}") no-repeat center center fixed;
            background-size: cover;
        }}
        header {{
            background-color: rgba(255, 255, 255, 0.5);
        }}
        body {{
            background-color: #FFEBEB; /* Light pink background color */
            font-family: 'Optima', sans-serif; 
            color: black;
        }}
        .stMarkdown {{
            background-color: rgba(255, 255, 255, 0.7); /* Light pink background for markdown text */
            color: black; /* Set text color to black */
            padding: 10px;
            border-radius: 10px;
            font-family: 'Optima', sans-serif; 
        }}
        .stTextInput {{
            background-color: rgba(255, 255, 255, 0.7); /* Light pink background for text input */
            border-radius: 10px;
            font-family: 'Optima', sans-serif;
            color: black;
        }}
        .stButton {{
            background-color: #FF69B4; /* Medium pink color for buttons */
            border-radius: 10px;
            font-family: 'Optima', sans-serif;
            color: black;
        }}
        .stRadio div div {{
            color: black; /* Set text color of radio buttons to black */
        }}
    </style>
"""

# Apply the background style using st.markdown
st.markdown(background_style, unsafe_allow_html=True)

# Open the laptop camera
cap = cv2.VideoCapture(0)  # Use the correct camera index (e.g., 0, 1, etc.)

# Check if the camera is opened successfully
if not cap.isOpened():
    st.error("Error: Could not open the camera. Please check if the camera is available.")
else:
    # Load the saved model
    model = load_model('models/shadesense_final_model3')

    # Load the label encoder classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('models/label_encoder_classes.npy', allow_pickle=True)

    # Streamlit app
    st.markdown(
        """
        <div style="text-align: center; font-size: 70px; font-family: 'Optima', sans-serif; color: black;">
            ShadeSense
        </div>
        """,
        unsafe_allow_html=True
    )

# Add custom CSS to style the text box
st.markdown(
    """
    <style>
        .custom-background {{
            padding: 10px; /* Add padding for better appearance */
            border-radius: 10px; /* Add rounded corners */
        }}

        .custom-text {{
            color: black; /* Set text color */
            font-size: 25px; /* Set font size */
            font-family: 'Optima', sans-serif; /* Set font style to Optima */
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Add text box with custom styling above radio button
st.markdown("<div class='custom-background'><div class='custom-text'>Upload an image or activate Live Detection</div></div>", unsafe_allow_html=True)

# Add two buttons for user choice
option = st.radio("Select an option:", ["Upload Image", "Live Detection"], key="radio_option", index=None)



if option is not None:
    if option == "Upload Image":
        # Section for Image Upload
        st.markdown(
            """
            <div style="text-align: center; font-size: 30px; font-family: 'Optima'; color: black;">
                Upload an image for Lipstick Detection
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Read the image file
            image = Image.open(uploaded_file)

            # Convert the image to RGB
            rgb_image = np.array(image.convert('RGB'))

            # Resize the image to match the model input size
            resized_image = cv2.resize(rgb_image, (512, 512))

            # Normalize the color by dividing pixel values by 255
            normalized_image = resized_image / 255.0

            # Display the uploaded image
            st.image(normalized_image, channels="RGB", use_column_width=True, caption="Uploaded Image")

            # Make prediction using the loaded model
            prediction = model.predict(np.expand_dims(normalized_image, axis=0))
            predicted_class_index = np.argmax(prediction)
            predicted_class = label_encoder.classes_[predicted_class_index]

            # Display the predicted class
            st.markdown(
                f"""
                <div style="text-align: center; font-size: 40px; font-family: 'Optima', cursive; color: black;">
                    Lipstick Shade: <span style="color: black; font-size: 50px;">{predicted_class}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif option == "Live Detection":
        # Center and style "Live Lipstick Detection" below it with black color
        st.markdown(
            """
            <div style="text-align: center; font-size: 30px; font-family: 'Optima'; color: black;">
                Live Lipstick Detection
            </div>
            """,
            unsafe_allow_html=True
        )

        # Create an empty st.image element
        image_element = st.empty()

        # Create an empty st.text element for displaying the predicted class
        predicted_class_element = st.text("Predicted Lipstick Shade: ")

        # Set the frequency for making predictions (e.g., every 2 seconds)
        prediction_frequency = 2  # in seconds
        last_prediction_time = time.time()

        # Continuously capture frames from the camera and update the image and text in-place
        while True:
            # Retrieve a frame from the camera
            ret, frame = cap.read()

            # Check if the frame retrieval is successful
            if not ret:
                st.error("Error: Could not retrieve frame from the camera.")
                break

            # Resize the frame to match the model input size
            frame = cv2.resize(frame, (512, 512))

            # Convert the frame to RGB (OpenCV uses BGR by default)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Normalize the color by dividing pixel values by 255
            normalized_frame = rgb_frame / 255.0

            # Update the image in-place
            image_element.image(normalized_frame, channels="RGB", use_column_width=True)

            # Check if it's time to make a prediction
            if time.time() - last_prediction_time > prediction_frequency:
                # Expand dimensions to make it a batch of 1
                image = np.expand_dims(normalized_frame, axis=0)

                # Make prediction using the loaded model
                prediction = model.predict(image)
                predicted_class_index = np.argmax(prediction)
                predicted_class = label_encoder.classes_[predicted_class_index]

                # Display the predicted class in a single space
                predicted_class_element.text(f"You're wearing: {predicted_class}")

                # Display the predicted class in a single space with custom style
                predicted_class_element.markdown(
                    f"""
                    <div style="text-align: center; font-size: 40px; font-family: 'Optima', cursive; color: hot pink;">
                        You're wearing: <span style="color: black; font-size: 50px;">{predicted_class}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Update the last prediction time
                last_prediction_time = time.time()

# Path to the feedback log file
log_file_path = "feedback_log.txt"

# Create the log file if it doesn't exist
with open(log_file_path, "a+") as log_file:
    pass

# Function to write feedback to the log file
def write_feedback(feedback):
    with open(log_file_path, "a") as log_file:
        log_file.write(feedback + "\n")


# Placeholder for the actual prediction 
# For demonstration purposes, using a list of possible shades
possible_shades = ["Crème D'Nude", "Honey Love", "Lasting Passion", "None", "Ruby Woo", "Stone", "Whirl"]
predicted_shade_index = 4  # Index representing the initial prediction
predicted_shade = possible_shades[predicted_shade_index]


# Add custom CSS to style the text box
st.markdown(
    """
    <style>
        .custom-background {
            padding: 10px; /* Add padding for better appearance */
            border-radius: 10px; /* Add rounded corners */
        }

        .custom-text {
            color: black; /* Set text color */
            font-size: 25px; /* Set font size */
            font-family: 'Optima', cursive; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add text box with custom styling above radio button
st.markdown("<div class='custom-background'><div class='custom-text'>Did we get it right?</div></div>", unsafe_allow_html=True)




# Create buttons for feedback with default value set to None
feedback = st.radio("Was the prediction accurate?", ["Thumbs Up", "Thumbs Down"], key="feedback_radio", index=None, help="color: black;")



# Process feedback
if feedback is not None:
    if feedback == "Thumbs Up":
        st.write("Thank you for your positive feedback!")
        # Save positive feedback to the log file
        write_feedback("Positive Feedback: Predicted Shade - {}".format(predicted_shade))
    else:
        st.write("Thank you for your feedback! We'll use this information to improve.")
        # Save negative feedback to the log file
        write_feedback("Negative Feedback: Predicted Shade - {}".format(predicted_shade))



    # Release the camera when done
    cap.release()