import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import glob


# Defining Flask App
app = Flask(__name__)
app.secret_key = '12345678'


nimgs = 100

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-Period1-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-Period1-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    df = pd.read_csv(latest_attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    df = pd.read_csv(latest_attendance_file)
    if int(userid) not in list(df['Roll']):
        with open(latest_attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder 
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)




################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    # Find the latest attendance file
    latest_attendance_file = max(glob.glob('Attendance/Attendance-*.csv'), key=os.path.getctime)

    # Extract attendance information from the latest file
    df = pd.read_csv(latest_attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)

    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/'+duser)

    ## if all the face are deleted, delete the trained file...
    if os.listdir('static/faces/')==[]:
        os.remove('static/face_recognition_model.pkl')
    
    try:
        train_model()
    except:
        pass

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)

    # Initialize default parameter values
    scaleFactor = 10
    minNeighbors = 5
    minSize = 20

    # Create the 'Attendance' window
    cv2.namedWindow('Attendance')

    # Create a separate window for sliders
    cv2.namedWindow('Sliders')
    cv2.moveWindow('Sliders', 20, 20)  # Move the sliders window to a specific position

    # Create trackbars for parameter adjustment
    cv2.createTrackbar('Scale Factor', 'Sliders', scaleFactor, 20, lambda x: None)
    cv2.createTrackbar('Min Neighbors', 'Sliders', minNeighbors, 20, lambda x: None)
    cv2.createTrackbar('Min Size', 'Sliders', minSize, 100, lambda x: None)

    while True:
        ret, frame = cap.read()

        # Get parameter values from sliders
        scaleFactor = max(1.1, cv2.getTrackbarPos('Scale Factor', 'Sliders') / 10)  # Ensure scaleFactor is greater than 1
        minNeighbors = cv2.getTrackbarPos('Min Neighbors', 'Sliders')
        minSize = cv2.getTrackbarPos('Min Size', 'Sliders')

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(minSize, minSize))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Attendance', frame)

        # Check for ESC key press to exit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']

    # Check if a user with the same ID already exists
    existing_users = os.listdir('static/faces')
    for user in existing_users:
        _, existing_userid = user.split('_')
        if existing_userid == newuserid:
            flash('User with the same ID already exists. Please choose a different ID or delete the existing user.')
            return redirect(url_for('home'))

    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Route to create a new period (new attendance file) and clear the attendance list
@app.route('/newperiod', methods=['GET'])
def newperiod():
    # Increment the period number
    period_number = len([file for file in os.listdir('Attendance') if file.startswith('Attendance-')])

    # Create a new attendance file for the new period
    new_attendance_file = f'Attendance/Attendance-Period{period_number + 1}-{datetoday}.csv'
    with open(new_attendance_file, 'w') as f:
        f.write('Name,Roll,Time')

    # Redirect back to the home page
    return redirect(url_for('home'))

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)