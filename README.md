Face Recognition Based Automatic Attendance System (FRBAAS)
.................................................

A step-by-step guide to run the Face Recognition Based Automatic Attendance System (FRBAAS) on a new user's desktop:

Prerequisites
Before you begin, ensure you have the following installed on your machine:

Python,

Git (optional)

Step 1: Clone the Repository
Open your terminal and run the following commands:
git clone https://github.com/ThinkerBytes/FRBAAS.git
cd FRBAAS

(if you are not using git, you can download this repository: https://github.com/ThinkerBytes/FRBAAS
)

Step 2: Create and Activate a Virtual Environment
On Windows:
python -m venv venv

venv\Scripts\activate

On macOS/Linux:
python3 -m venv venv

source venv/bin/activate

Step 3: Install Dependencies
pip install -r requirements.txt

once the requiremets is installed

Step 4: Run the Flask App
python app.py

(The Flask app will start running. Access it in your web browser at http://127.0.0.1:5000/.)

Step 5: Explore the App:
To add a new user, enter the name and ID, and click "Add New User."
Click on "Take Attendance" to start the face recognition process.
(A python camera interface will openup.)
Press the esc button when done.
View the list of attendees on the main page.

.......................................................

Feature to be added.
1. Improved ui/ux
2. Responsive Design across device
3. List user feature were all the user in the database is listed.
4. Delete user feature, where a user and all his data can be deleted.
5. Implement Period wise attendence.
6. Improve ML model accuracy.


........................................................

Detailed explanation of the web app.


This Flask web application implements a Face Recognition Based Automatic Attendance System. Here's a short description of how the app works:

Web Interface:

The app has a simple web interface with two main sections:
Today's Attendance: Displays the current attendance list, including the name, ID, and time of each attendee.
Add New User: Allows the addition of a new user by entering their name and ID.
Today's Attendance:

Clicking on the "Take Attendance" button starts the face recognition process.
The webcam captures frames, and the app detects faces using Haar cascades.
Identified faces are recognized using a pre-trained machine learning model (K-Nearest Neighbors classifier).
The detected face's name and ID are added to the attendance list, along with the current time.
Pressing the 'esc' key stops the face recognition process, and the attendance list is displayed on the web interface.
Add New User:

Users can add a new person to the system by entering their name and ID.
When adding a new user, the webcam opens to capture multiple images of the person's face (controlled by nimgs variable).
The captured face images are stored in the "static/faces" directory with a unique folder for each user.
The machine learning model is retrained using the new data to recognize the added user.
Training Model:

The machine learning model (K-Nearest Neighbors) is trained on all the face images stored in the "static/faces" directory.
The trained model is saved as "static/face_recognition_model.pkl" for later use in face recognition.
Attendance Log:

The app maintains an attendance log in the "Attendance" directory, storing attendance records for each day in a CSV file.

Additional Features:

The web interface provides information on the total number of registered users.
Dependencies:

The app uses various libraries, including Flask for the web framework, OpenCV for face detection, scikit-learn for machine learning, and pandas for data manipulation.
Running the App:

The Flask app runs locally, and the user interacts with it through a web browser.
The app should be executed from the command line (python app.py), and the web interface can be accessed at http://127.0.0.1:5000/ in the browser.




How face Recognition works:

Haar Cascade Classifier for Face Detection
1. Haar Cascade Classifier:
Haar Cascade is a machine learning object detection method used to identify objects in images or video. It works based on the Haar Wavelet technique to identify features.

2. Face Detection:
The cv2.CascadeClassifier in OpenCV is a pre-trained Haar Cascade classifier designed for face detection. It's a set of XML files containing feature definitions for various objects, including faces. You are using the file haarcascade_frontalface_default.xml.

3. detectMultiScale Function:
The detectMultiScale function is a key component of Haar Cascade face detection. It detects objects in the image by moving a window over the image at different scales and using the Haar-like features to identify the object.

scaleFactor: Specifies how much the image size is reduced at each image scale. A lower value detects smaller faces but increases computation.

minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it. Higher values result in fewer detections but with higher quality.

minSize: Objects below this size are ignored.

4. Extracting Faces:
In your extract_faces function, you convert the image to grayscale and then apply the detectMultiScale function. If faces are detected, their coordinates (x, y, width, height) are returned.

5. Face Recognition Model:
After extracting the faces, you use a machine learning model for face recognition. You've implemented the k-Nearest Neighbors (KNN) algorithm. This model is trained on the faces stored in the static/faces directory.

6. identify_face Function:
This function takes the flattened array of a face and uses the trained KNN model to predict the identity.

7. Adding Attendance:
In the start route, you continuously capture frames from the camera, detect faces, recognize them, and add their attendance to the CSV file.

Improving Multiple Face Detection:
The issue you faced with detecting multiple faces could be due to the parameters in detectMultiScale. Adjusting parameters like scaleFactor, minNeighbors, and minSize might help in detecting multiple faces. You've already made some adjustments in the extract_faces function.

Remember that lighting conditions, camera angles, and the distance of faces from the camera can affect detection. Experiment with these parameters to optimize face detection for your specific setup.


