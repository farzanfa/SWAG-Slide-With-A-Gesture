# TechVidvan hand Gesture Recognizer

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from face_img import predict1

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
#=========
from tkinter.filedialog import askopenfilename
import win32com.client
import time
from audio import takeCommand

app = win32com.client.Dispatch("PowerPoint.Application")

#file=askopenfilename()
presentation = app.Presentations.Open(FileName='C:/Users/farza/Downloads/gesture_ppt/python1.ppt', ReadOnly=1)

flag1=False
flag2=False
change_count=0
start_flag=False

#========
# Initialize the webcam


while True:
    print(flag1,'flag1')
    print(flag2,'flag2')
    # Read each frame from the webcam
    if start_flag==False:
        cap = cv2.VideoCapture(0)
        start_flag=True
    if flag2==False:
        
        _, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if flag2==False:
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)
        print(className,'==')               

        # Show the final output
        names,frame1=predict1(frame)
        names=set(names)
        print(names)
    

        cv2.imshow("Output", frame) 
        cv2.waitKey(1)

        if 'person' not in names:
            continue

    if flag2==False:

        if className=='7':
            if flag1==False:
                try:
                    presentation.SlideShowSettings.Run()
                    flag1=True
                except:
                    pass  
        else:
            if flag1==False:
                pass
            else:
                if className=='5':
                    try:
                        presentation.SlideShowWindow.View.Next()
                        time.sleep(1)
                    except:
                        flag1=False
                        pass    
                elif className=='4':
                    try:
                        presentation.SlideShowWindow.View.Previous()
                        time.sleep(1) 
                    except:
                        flag1=False
                        pass       
                elif className=='6':
                    try:
                        presentation.SlideShowWindow.View.Exit()    
                    except:
                        flag1=False
                        pass    
                elif className=='2':
                    change_count+=1
                    print(change_count)
                    if change_count>10:
                        flag2=True
                        change_count=0   
                        cap.release() 
                        
                        cv2.destroyAllWindows()
    else:
        q=takeCommand()
        if q=='start':
            if flag1==False:
                pass
            try:
                presentation.SlideShowSettings.Run()
                flag1=True
            except:
                flag1=False
                pass    
        else:
            if flag1==False:
                pass
            else:
                if q=='right':
                    try:
                        presentation.SlideShowWindow.View.Next()
                        time.sleep(1)
                    except:
                        flag1=False
                        pass  
                elif q=='left':
                    try:
                        presentation.SlideShowWindow.View.Previous()
                        time.sleep(1) 
                    except:
                        pass    
                elif q=='exit':
                    try:
                        presentation.SlideShowWindow.View.Exit()    
                    except:
                        flag1=False
                        pass 
                elif q=='change':
                    cap = cv2.VideoCapture(0)
                    flag2=False                        




    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()