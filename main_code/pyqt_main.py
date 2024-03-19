# Step 0: Install packages.

#!pip install PyQt5
#!pip install opencv-python
#!pip install mediapipe
#!pip install ultralytics
#!pip install opencv-contrib-python

# Step 1: Initialise the Splashscreen to execute the subsequent code prior to displaying the Ui_MainWindow (Step 9).

import os
import time
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

current_dir = os.getcwd()
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) #Create Splashscreen for program loading
    splash_object = QtWidgets.QSplashScreen(QtGui.QPixmap(current_dir+"\sitting_vector.png"))

    # Center the program to the middle of monitor 0
    monitor_num = 0
    monitor = QtWidgets.QDesktopWidget().screenGeometry(monitor_num)
    splash_width = splash_object.size().width()
    splash_height = splash_object.size().height()
    splash_object.move(monitor.center().x()-int(splash_width/2), monitor.center().y()-int(splash_height/2))
    splash_object.setWindowTitle(u'Program Loading...')

    # Show splashscreen on top of window at first, but NOT ALWAYS on top
    splash_object.setWindowFlags(splash_object.windowFlags() | Qt.WindowStaysOnTopHint)
    splash_object.setWindowState(splash_object.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
    splash_object.setWindowFlags(splash_object.windowFlags() & ~Qt.WindowStaysOnTopHint)
    splash_object.raise_()
    splash_object.show()
    splash_object.activateWindow()

# Step 2: Import the remaining necessary libraries.

import mediapipe as mp
from ultralytics import YOLO
import math
from math import atan2, degrees, acos
import cv2
import shutil
import numpy as np
import sys
import keras
from keras.models import load_model
import winsound
import warnings

# Step 3: Initialise some essential global variables, these variables might be changed in later part.

# Used for step 9 output on textbrowser
texts = [" "," "," "]

# Used for 'settings' of the program in Step 8
sit_side = True
sco_check = True
eye_check = True
drowsy_check = True
kyp_check = True
body_check = True
head_check = True
sc_alarm = 100
open_close_value = 0.9

# Used for alarming for the drowsiness detection in Step 5
first_checker = True

# Step 4: Define math functions part 1 (For Front Detection Part)


# used for finding the midpoint of between two points of landmark from mediapipe
def midpoint(p1, p2, width, height):
    return (p1.x+p2.x)/2 * width, (p1.y+p2.y)/2 * height

# used for finding the gradient of between two points of landmark from mediapipe
def slope(p1, p2, width, height):
    m = (p2.y - p1.y)* height /((p2.x - p1.x)* width)
    return m

# used for finding the constant c of normal y=mx+c that passes the midpoint of p1 and p2
def normal_c(p1, p2, width, height):
    p1 = p1.x * width, p1.y * height
    p2 = p2.x * width, p2.y * height
    return ((p1[1] ** 2 - p2[1] ** 2) + (p1[0] ** 2 - p2[0] ** 2))/(2*(p1[1] - p2[1]))

# used for finding the included angle between three points
def angle_between_scoliosis(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else (deg1 - deg2)

# set the critiria for the alert output when receive the angle of  scoliosis angle
def scoliosis_alert(x):
    if x<=10:
      return "good" ,(50,205,50)
    elif x>10 and x<=25:
      return "mild" ,(255,234,0)
    elif x>25 and x<=45:
      return "moderate" ,(255,172,28)
    elif x>45:
      return "severe" ,(255,0,0)
    else:
      return "error",(0,0,0)

# given the m and c of two straight lines (y = mx +c) find the intersection point
def line_intersect_gradient(m1, c1, m2, c2):
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y

# return true if num0 falls between num1 and num2
def btw_range(num0, num1, num2):
    return num1<=num0<=num2 or num2<=num0<=num1

# given the top left and bottom right points of rectangules,
# return true if two rectangular are overlap
def rect_overlap(x01, y01, x02, y02, x11, y11, x12, y12):
    return not (x02 < x11
                or x01 > x12
                or y02 < y11
                or y01 > y12)

# Step 5: Front Detection and Analysis

# detect front camera

# initialize the parameter and setup for medaipipe landmark detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=0,
                    smooth_landmarks=True,
                    enable_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


# import haar cascade classifier for left and right eyes
leftEyeCas = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml'))
rightEyeCas = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml'))

# import the pre-trained open eye close eye detection model
drowsiness_model = load_model(current_dir+'/trained model/real_eyes.h5')

# initialize other usable variables

# image resize for open/close eye detection, because the model is trained under 224*224 images
resize_size=(224,224)

# use score to record the current Cumulative score for drowsiness detection,
# activate alarm when surpass the set 'sc_alarm' in 'settings'
score=0

# frequency and duration of drowsiness detection alarm
freq=400
dur=250

# function return the real coordinate of the points from mediapipe
def mdp_real(coor,width,height):
    return coor.x * width,coor.y * height

# mediapipe + eye detection (front detection part)
def front_detect(img):

    # initialise
    results = pose.process(img)
    height, width, channels = img.shape
    color_img = img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sco_angle = "     "

    # scoliosis tracking
    try:
        # get the midpoints of both eye and shoulder landmarks (real coordinate)
        eye_mid = midpoint(results.pose_landmarks.landmark[1], results.pose_landmarks.landmark[4], width, height)
        shoulder_mid = midpoint(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12], width, height)

        # get the constant value c from the y=mx+c,
        # aka the straight line that passes the midpoint of shoulder and normal of points of shoulder landmarks
        shoulder_c = normal_c(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12], width, height)
        eye_grad = slope(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12], width, height)

        # get gradient m of shoulder(neck) and eye, and value c of y=mx+c of eye straight line
        shoulder_grad = -1 / eye_grad
        eye_c = eye_mid[1] - eye_grad * eye_mid[0]

        # get the intersection points
        eye_horizon = line_intersect_gradient(eye_grad, eye_c, shoulder_grad, shoulder_c)

        # get the scoliosis angle
        scoliosis_angle = angle_between_scoliosis(eye_mid, shoulder_mid, eye_horizon)

        # export and modify the computed result
        P1, P2, P3 = eye_mid, shoulder_mid, eye_horizon

        # to ensure that the part for scoliosis angle only occupy 5 empty spaces
        sco_angle=str(round(scoliosis_angle,1))
        while len(sco_angle) < 5:
            sco_angle=" "+sco_angle

        # get the color and output for scoliosis detection based on the function in step 4
        scoliosis_output,scoliosis_rgb = scoliosis_alert(scoliosis_angle)

        # if the checkbox of scoliosis detection in 'settings' is checked, draw line and points on image
        if sco_check:
            img = cv2.line(img,(int(P1[0]),int(P1[1])),(int(P2[0]),int(P2[1])),(128,0,0),2)
            img = cv2.line(img,(int(P2[0]),int(P2[1])),(int(P3[0]),int(P3[1])),(255,0,0),2)
            img = cv2.line(img,(int(P1[0]),int(P1[1])),(int(P3[0]),int(P3[1])),(255,255,255),2)
            img = cv2.circle(img,(int(P1[0]),int(P1[1])),0,(0,0,255),3)
            img = cv2.circle(img,(int(P2[0]),int(P2[1])),0,(0,0,255),3)
            img = cv2.circle(img,(int(P3[0]),int(P3[1])),0,(0,0,255),3)

    except:
        # when some of the required body landmark cannot detected
        scoliosis_output,scoliosis_rgb = 'No body detected',(255,255,255)

    # if the checkbox of scoliosis detection in 'settings' is checked, add text image
    if sco_check:
        img = cv2.putText(img,scoliosis_output,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5,cv2.LINE_AA)
        img = cv2.putText(img,scoliosis_output,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,scoliosis_rgb,2,cv2.LINE_AA)

    # another initialise
    right_status = "     "
    left_status = "     "
    eye_output = "Cannot detect"
    eye_corner = ""
    global score
    global first_checker

    try:
        # get the real coordinate where eyes locate, and use haar cascade detector to frame them
        reye_coor = mdp_real(results.pose_landmarks.landmark[5], width, height)
        leye_coor = mdp_real(results.pose_landmarks.landmark[2], width, height)

        left_eye = leftEyeCas.detectMultiScale(gray)
        right_eye = rightEyeCas.detectMultiScale(gray)

        # left eye part,
        # using the pretrained open/close eye model to detect only the cropped part from the haar cascade detector
        try:
            for (lx,ly,lw,lh) in left_eye:
                if btw_range(leye_coor[0],lx,lx+lw) and btw_range(leye_coor[1],ly,ly+lh):
                    if eye_check:
                        img = cv2.rectangle(img,(lx,ly),(lx+lw,ly+lh),(255,255,255),2)

                    l_eye = color_img[ly:ly+lh, lx:lx+lw]
                    l_eye = cv2.resize(l_eye, resize_size)
                    l_eye = l_eye.reshape((1, 224, 224, 3))
                    lpred = drowsiness_model.predict(l_eye,verbose = 0)

                    eye_output = "right eye detect"#switch the left and right since the img is inverted

                    if(lpred[0][0] < open_close_value):
                        left_status = "open "
                        score-=1
                        break
                    else:
                        left_status = "close"
                        score+=1
                        break
        except:
            pass

        # right eye part,
        # using the pretrained open/close eye model to detect only the cropped part from the haar cascade detector
        try:
            for (rx,ry,rw,rh) in right_eye:
                if btw_range(reye_coor[0],rx,rx+rw) and btw_range(reye_coor[1],ry,ry+rh):
                    if eye_check:
                        img = cv2.rectangle(img,(rx,ry),(rx+rw,ry+rh),(255,255,255),2)

                    r_eye = color_img[ry:ry+rh, rx:rx+rw]
                    r_eye = cv2.resize(r_eye, resize_size)
                    r_eye = r_eye.reshape((1, 224, 224, 3))
                    rpred = drowsiness_model.predict(r_eye,verbose = 0)

                    if eye_output == "right eye detect":
                        eye_output = "both eye detect"
                    else:
                        eye_output = "left eye detect"#switch the left and right since the img is inverted

                    if(rpred[0][0] < open_close_value):
                        right_status = "open "
                        score-=1
                        break
                    else:
                        right_status = "close"
                        score+=1
                        break
        except:
            pass

        # if dowsiness detection is checked in settings,
        # set range of score to 0 <= score <= sc_alarm*2,
        # while output as drowsy when score > sc_alarm
        if drowsy_check:
            if(score<0):
                score=0
            elif(score>sc_alarm*2):
                score=sc_alarm*2
                eye_corner = "drowsy"
            elif(score>sc_alarm):
                eye_corner = "drowsy"
    except:
        pass

    # if dowsiness detection is checked in settings, alarm when score > sc_alarm at first time,
    # reset the first time when score < sc_alarm
    if drowsy_check:
        if eye_corner == "drowsy" and first_checker:
            winsound.Beep(freq,dur)
            winsound.Beep(freq,dur)
            first_checker = False
        elif eye_corner != "drowsy" and not first_checker:
            first_checker = True

        # show text on image
        img = cv2.putText(img,eye_corner,(200,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5,cv2.LINE_AA)
        img = cv2.putText(img,eye_corner,(200,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

    # show text on image
    if eye_check:
        img = cv2.putText(img,eye_output,(10,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5,cv2.LINE_AA)
        img = cv2.putText(img,eye_output,(10,220),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

    # compile the output text
    text1 = "("+right_status+","+left_status+")"

    # return for usage of Ui_MainWindow in step 9
    return img, text1, sco_angle

# Step 6: Define math functions part 2 (For Side Detection Part)

# set the critiria for the alert output when receive the angle of kyphosis angle
def kyphosis_alert(x):
    if x<=50:
      return "good" ,(50,205,50) #green color
    elif x>50:
      return "bad" ,(255,0,0) #red color
    else:
      return "error",(0,0,0)

# get the circle defined by three points, return the radius and center of circle,
# and the angle of pt2 and pt3 located
def get_circle(pt1, pt2, pt3):
    sqrt = math.sqrt

    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3

    l1 = (x1 - x3) ** 2 + (y1 - y3) ** 2

    s1 = x1**2 + y1**2
    s2 = x2**2 + y2**2
    s3 = x3**2 + y3**2
    M11 = x1*y2 + x2*y3 + x3*y1 - (x2*y1 + x3*y2 + x1*y3)
    M12 = s1*y2 + s2*y3 + s3*y1 - (s2*y1 + s3*y2 + s1*y3)
    M13 = s1*x2 + s2*x3 + s3*x1 - (s2*x1 + s3*x2 + s1*x3)
    x0 =  0.5*M12/M11
    y0 = -0.5*M13/M11
    r0 = sqrt((x1 - x0)**2 + (y1 - y0)**2)


    pt2_angle = 180*math.atan2(y2 - y0, x2 - x0)/math.pi
    pt3_angle = 180*math.atan2(y3 - y0, x3 - x0)/math.pi

    if(pt2_angle>=pt3_angle):
        pt2_angle=pt2_angle-360

    return (x0, y0), r0, pt2_angle, pt3_angle

# get the angle occupied by small radius r1 (head radius) and large radius r0 (radius get in get_circle)
def cal_sector_angle(r0, r1):
    return math.degrees(math.acos((2*r0**2 - r1**2)/(2.0*r0**2)))

# draw the arc on image
def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=4, lineType=cv2.LINE_AA, shift=10):

    center = (
        int(round(center[0] * 2**shift)),
        int(round(center[1] * 2**shift))
    )
    axes = (
        int(round(axes[0] * 2**shift)),
        int(round(axes[1] * 2**shift))
    )
    return cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)

# Step 7: Side Detection and Analysis

# detect side camera
# import pretrained model of head and body detection
body_model = YOLO(current_dir+'/trained model/best_body.pt')
threshold = 0.5

# Side detection: yolov8 + computation
def side_detect(img):

    # initialise
    results = body_model(img, verbose=False)[0]
    head_data = []
    body_data = []
    class_list = []
    kyp_angle = "     "

    # detect only 1 head nad 1 body at most,
    # get the rectangle frame of head and body part,
    # activate kyphosis detection when both head and body are detected
    # in kyphosis detection, compute and draw the spine angle using the function in step 6

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold and class_id not in class_list:
            class_list.append(class_id)
            if results.names[int(class_id)] == "head":#head
                default_rgb = (255,255,255)
                spn_up = ((x1+x2)/2,(y1+y2)/2)
                head_radius = min((x1-x2)/2,(y1-y2)/2)
                head_y = y2-y1
            else:
                default_rgb = (0, 255, 0)
                spn_down = x1,y2
                body_y = y2-y1

            if default_rgb == (0, 255, 0) and body_check:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), default_rgb, 4)
                img = cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, default_rgb, 2, cv2.LINE_AA)
            elif default_rgb == (255, 255, 255) and head_check:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), default_rgb, 4)
                img = cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, default_rgb, 2, cv2.LINE_AA)

            if len(class_list) == 2 and (body_y>2*head_y):

                temp_coor = (spn_up[0], spn_down[1] *2 - spn_up[1])
                center, radius, start_angle, end_angle = get_circle(temp_coor,spn_down,spn_up)


                sector_angle = cal_sector_angle(radius,head_radius)
                end_angle = end_angle - sector_angle
                axes = (radius, radius)
                if kyp_check:
                    img = draw_ellipse(img, center, axes, 0, start_angle, end_angle, 255)

                kyphosis_angle = end_angle - start_angle

                # to ensure that the part for scoliosis angle only occupy 5 empty spaces
                kyp_angle=str(round(kyphosis_angle,1))
                while len(kyp_angle) < 5:
                    kyp_angle=" "+kyp_angle


                kyphosis_output,kyphosis_rgb = kyphosis_alert(kyphosis_angle)

                # when kyphosis detection in settings is checked, put text on image
                if kyp_check:
                    img = cv2.putText(img,kyphosis_output,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),5,cv2.LINE_AA)
                    img = cv2.putText(img,kyphosis_output,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,kyphosis_rgb,2,cv2.LINE_AA)

    # return for usage of Ui_MainWindow in step 9
    return img, kyp_angle

# Step 8: GUI setup for 'settings'

# ignore DeprecationWarning as it is not an error but frequently appear in some python version, filter them
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Layout for 'settings'
class Ui_Dialog(QtWidgets.QDialog):
    def setupUi(self):

        self.resize(500, 300)
        self.setFixedSize(QtCore.QSize(500, 300))
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 10, 471, 271))

        self.gridLayout = QtWidgets.QGridLayout(self.horizontalLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout_2 = QtWidgets.QVBoxLayout()

        self.verticalLayout_4 = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget)

        self.verticalLayout_4.addWidget(self.label)

        self.checkBox_2 = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_2.setChecked(sco_check)
        self.verticalLayout_4.addWidget(self.checkBox_2)

        self.checkBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox.setChecked(eye_check)
        self.verticalLayout_4.addWidget(self.checkBox)

        self.checkBox_6 = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_6.setChecked(drowsy_check)
        self.verticalLayout_4.addWidget(self.checkBox_6)

        self.verticalLayout_2.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()

        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.verticalLayout_5.addWidget(self.label_2)

        self.checkBox_5 = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_5.setChecked(kyp_check)
        self.verticalLayout_5.addWidget(self.checkBox_5)

        self.checkBox_4 = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_4.setChecked(body_check)
        self.verticalLayout_5.addWidget(self.checkBox_4)

        self.checkBox_3 = QtWidgets.QCheckBox(self.horizontalLayoutWidget)
        self.checkBox_3.setChecked(head_check)
        self.verticalLayout_5.addWidget(self.checkBox_3)

        self.verticalLayout_2.addLayout(self.verticalLayout_5)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()

        self.verticalLayout_6 = QtWidgets.QVBoxLayout()

        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.side_combo= QtWidgets.QComboBox()
        self.side_combo.addItems(["User\'s left","User\'s right"])
        self.side_combo.setCurrentIndex(1-int(sit_side))

        self.verticalLayout_6.addWidget(self.label_3)

        self.verticalLayout_6.addWidget(self.side_combo)
        self.verticalLayout.addLayout(self.verticalLayout_6)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()

        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget)

        self.verticalLayout_7.addWidget(self.label_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()

        self.spinBox = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.spinBox.setMinimum(1)
        self.spinBox.setValue(int(100 - open_close_value*100))

        self.horizontalLayout_5.addWidget(self.spinBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()

        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget)

        self.verticalLayout_8.addWidget(self.label_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()

        self.spinBox_2 = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.spinBox_2.setMinimum(1)
        self.spinBox_2.setValue(int(100-sc_alarm/10))

        self.horizontalLayout_4.addWidget(self.spinBox_2)
        self.label_6 = QtWidgets.QLabel(self.horizontalLayoutWidget)

        self.horizontalLayout_4.addWidget(self.label_6)
        self.verticalLayout_8.addLayout(self.horizontalLayout_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()

        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setDefault(True)

        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)

        self.horizontalLayout.addWidget(self.pushButton_2)
        self.verticalLayout_8.addLayout(self.horizontalLayout)
        self.verticalLayout.addLayout(self.verticalLayout_8)
        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)

        self.setWindowTitle("Settings")
        self.label.setText("Front view")
        self.checkBox_2.setText("Show scoliosis visualisation")
        self.checkBox.setText("Show eye detection box")
        self.checkBox_6.setText("Enable drowsiness detection")
        self.label_2.setText("Side View")
        self.checkBox_5.setText("Show kyphosis visualisation")
        self.checkBox_4.setText("Show body detection box")
        self.checkBox_3.setText("Show head detection box")
        self.label_3.setText("Webcam/2nd Cam is located at")
        self.label_4.setText(" Opened eye size (1 - 99)")
        self.label_5.setText("Drowsiness detector sensitivity")
        self.label_6.setText("(1 - 99)")
        self.pushButton_3.setText("Save settings")
        self.pushButton_2.setText("Cancel")

        self.button_clicked()



        QtCore.QMetaObject.connectSlotsByName(self)

    # Define event, when the question mark button on pyqt is pressed
    def event(self, event):
        if event.type() == QtCore.QEvent.EnterWhatsThisMode:
            QtWidgets.QMessageBox.about(self, "Help",
                                    "This is a window for the help of "
                                    "this GUI system settings")
            return True
        return QtWidgets.QDialog.event(self, event)

    # Define function to connect when the 'save' or 'cancel' buttons is selected
    # When 'cancel' button is selected, do quit
    def button_clicked(self):
        self.pushButton_3.clicked.connect(self.saveit)
        self.pushButton_2.clicked.connect(self.reject)

    # When save button is pressed, update the global variables in step 3, then quit
    def saveit(self):
        global sit_side
        global open_close_value
        global sco_check
        global eye_check
        global drowsy_check
        global kyp_check
        global body_check
        global head_check
        global sc_alarm

        sco_check = self.checkBox_2.isChecked()
        eye_check = self.checkBox.isChecked()
        drowsy_check = self.checkBox_6.isChecked()
        kyp_check = self.checkBox_5.isChecked()
        body_check = self.checkBox_4.isChecked()
        head_check = self.checkBox_3.isChecked()

        combotext = self.side_combo.currentText()
        if combotext == "User\'s left":
            left_check = True
        else:
            left_check = False

        sit_side = left_check

        close_bound = self.spinBox.value()
        open_bound = 100 - close_bound
        open_close_value = open_bound/100

        sc_alarm = (100-self.spinBox_2.value())*10

        self.reject()

# Step 9: GUI setup for 'Main (Ui_MainWindow)'

# Initialise the size of layout for Ui_MainWindow, this value cannot be changed
width = 828
height = 351

# Layout for 'Ui_MainWindow'
class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.timer_cameras = []
        self.caps = []
        self.CAM_NUMS = [0, 1]

        self.set_ui()
        self.slot_init()

    def set_ui(self):

        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)

        #self.activateWindow()
        #self.setWindowState(self..windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        #self.showNormal()

        font = QtGui.QFont()
        font.setFamily("courier")
        font.setPointSize(8)
        self.textBrowser = QtWidgets.QLabel("Click 'Camera On' to enable the system to detect body")
        self.textBrowser.setAlignment(Qt.AlignCenter)
        self.textBrowser.setFont(font)
        self.mm_layout =  QtWidgets.QVBoxLayout()
        self.l_down_widget = QtWidgets.QWidget()

        myFont=QtGui.QFont()
        myFont.setBold(True)

        self.camera_name1 = QtWidgets.QLabel("Camera 1 ( Front View )")
        self.camera_name1.setAlignment(Qt.AlignCenter)
        self.camera_name1.setFont(myFont)
        self.camera_name2 = QtWidgets.QLabel("Camera 2 ( Side View )")
        self.camera_name2.setAlignment(Qt.AlignCenter)
        self.camera_name2.setFont(myFont)

        self.labels_layout =  QtWidgets.QHBoxLayout()
        self.cameras_layout =  QtWidgets.QVBoxLayout()
        self.__layout_view = QtWidgets.QHBoxLayout()
        self.__layout_main = QtWidgets.QHBoxLayout()

        self.label_cameras = [QtWidgets.QLabel() for _ in range(2)]

        for label_camera in self.label_cameras:
            label_camera.setFixedSize(320, 240)
            label_camera.setStyleSheet('''QWidget{border-radius:7px;background-color:#d3d3d3;}''')

        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.button_open_cameras = QtWidgets.QPushButton('Camera On')
        self.button_reset = QtWidgets.QPushButton('Settings')
        self.button_help = QtWidgets.QPushButton('Help')
        self.button_close = QtWidgets.QPushButton('Close Window')
        self.button_open_cameras.setMinimumHeight(50)
        self.button_reset.setMinimumHeight(50)
        self.button_help.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        self.__layout_fun_button.addWidget(self.button_open_cameras)
        self.__layout_fun_button.addWidget(self.button_reset)
        self.__layout_fun_button.addWidget(self.button_help)
        self.__layout_fun_button.addWidget(self.button_close)

        for label_camera in self.label_cameras:
            self.__layout_view.addWidget(label_camera)

        self.labels_layout.addWidget(self.camera_name1)
        self.labels_layout.addWidget(self.camera_name2)

        self.cameras_layout.addLayout(self.__layout_view)
        self.cameras_layout.addLayout(self.labels_layout)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addLayout(self.cameras_layout)


        self.l_down_widget.setLayout(self.__layout_main)
        self.mm_layout.addWidget(self.textBrowser)
        self.mm_layout.addWidget(self.l_down_widget)
        self.setLayout(self.mm_layout)
        self.setWindowTitle(u'Sit Posture Recognition System')

        self.setFixedSize(width,height)

    # buttons connected to function when selected
    def slot_init(self):
        self.button_close.clicked.connect(self.realclose)
        self.button_reset.clicked.connect(self.open_dialog)
        self.button_help.clicked.connect(self.help_msg)
        self.set_camera()

    # define and initialise the setup for webcam
    def set_camera(self):
        self.button_open_cameras.clicked.connect(self.button_open_cameras_clicked)

        for i, cam_num in enumerate(self.CAM_NUMS):
            timer_camera = QtCore.QTimer()
            self.timer_cameras.append(timer_camera)

            cap = cv2.VideoCapture()
            self.caps.append(cap)

            self.timer_cameras[i].timeout.connect(lambda i=i, cam_num=cam_num: self.show_camera(i, cam_num))

    # check the cameras is active or not, if no give warning, if yes start streaming
    # text of button changed to 'camera off', if being pressed then 'do 'close_camera'
    def button_open_cameras_clicked(self):
        for i, cam_num in enumerate(self.CAM_NUMS):
            if not self.timer_cameras[i].isActive():
                flag = self.caps[i].open(cam_num, cv2.CAP_DSHOW)
                if not flag:
                    msg = QtWidgets.QMessageBox.warning(self, 'warning', f"Please check camera {cam_num} is connected or not",
                                                    buttons=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_cameras[i].start(30)
        self.button_open_cameras.setText('Camera off') if any(timer.isActive() for timer in self.timer_cameras) else self.button_open_cameras.setText('Camera On')
        self.button_open_cameras.clicked.disconnect()
        self.button_open_cameras.clicked.connect(self.close_camera)

    # close every active cameras, re-initialise some variables,change button text to 'camera on', reconnect to the button function define earlier
    def close_camera(self):
        for i, cam_num in enumerate(self.CAM_NUMS):
            if self.timer_cameras[i].isActive():
                self.caps[i].release()
                self.label_cameras[i].clear()

        global score
        score = 0


        self.timer_cameras=[]
        self.caps=[]
        self.button_open_cameras.setText('Camera On')
        self.button_open_cameras.clicked.disconnect()
        self.set_camera()


    # using the frames for webcam streaming to do detection in step 5 and step 7.
    def show_camera(self, index, cam_num):
        flag, image = self.caps[index].read()

        global texts
        try:
            if sit_side:
                image = cv2.flip(image, 1)
            show = cv2.resize(image, (320, 240))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)


            if cam_num == 0:
                show,texts[0],texts[1] = front_detect(show)
            elif cam_num == 1:
                show,texts[2] = side_detect(show)

            text_to_write = "Eyes status:"+texts[0]+" Scoliosis angle : "+texts[1]+" Kyphosis angle : "+texts[2]
            self.textBrowser.setText(text_to_write)


            show_image = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_cameras[index].setPixmap(QtGui.QPixmap.fromImage(show_image))
        except:
            pass

    # when the cancel button is pressed, deactivate webcam
    def realclose(self):
        for i, cam_num in enumerate(self.CAM_NUMS):
            if self.timer_cameras[i].isActive():
                self.caps[i].release()
        self.close()

    # when settings button is pressed, open the gui defined in step 8
    def open_dialog(self):
        set_dialog = Ui_Dialog()
        set_dialog.setupUi()
        set_dialog.show()
        self.button_reset.clicked.disconnect()
        if not set_dialog.exec_():
            self.button_reset.clicked.connect(self.open_dialog)

    # show message of help in  message box
    def help_msg(self):
        QtWidgets.QMessageBox.about(self, "Help",
                            ">  Ensure your cameras are properly positioned and securely mounted.\n"
                            ">  Good lighting conditions improve detection accuracy.\n"
                            ">  Once the cameras are on, the application will automatically "
                            "analyze your posture. Results will be displayed in the status bar at the top.\n"
                            ">  Accessing Settings: Click the 'Settings' button to open a dialog where you can "
                            "customize various aspects of the detection and visualization process, including:\n"
                            ">  Enabling/disabling specific visualizations (e.g., scoliosis, kyphosis, eye detection "
                            "boxes).\n"
                            ">  Adjusting detection sensitivities and parameters."
                            )

    # when the close button is pressed, deactivate webcam
    def closeEvent(self, event):
        for i, cam_num in enumerate(self.CAM_NUMS):
            if self.timer_cameras[i].isActive():
                self.caps[i].release()
        event.accept()

if __name__ == '__main__':

    ui = Ui_MainWindow()

    # move Ui_MainWindow to center of monitor 0,
    # close splashscreen in step 2 when Ui_MainWindow is shown
    ui.move(monitor.center().x() - int(width/2), monitor.center().y() - int(height/2))
    splash_object.finish(ui)

    # show Ui_MainWindow on top on screen at first, but NOT ALWAYS on top
    ui.setWindowFlags(ui.windowFlags() | Qt.WindowStaysOnTopHint)
    ui.setWindowState(ui.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
    ui.setWindowFlags(ui.windowFlags() & ~Qt.WindowStaysOnTopHint)
    ui.raise_()
    ui.show()
    ui.activateWindow()

    sys.exit(app.exec_())

sys.exit()