import face_recognition
import pickle
import cv2
import numpy as np
import cv2
import sys

encodings="encodings.pickle"
detection_method="cnn"
display=1

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(encodings, "rb").read())
prototxtPath =("deploy.prototxt")
weightsPath ="res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


def detect_face(frame,faceNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        #print(confidence)
        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # (startX, startY, endX, endY)=(startX, startY-50, endX, endY+50)
            # (startX, startY) = (max(0, startX), max(0, startY))
            # (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            locs.append((int(startY), int(endX), int(endY), int(startX)))  
            cv2.rectangle(frame, (int(startX),  int(startY)), (int(endX),int(endY)), (0, 0, 255), 2)   
    # only make a predictions if at least one face was detected
    
    # return a 2-tuple of the face locations and their corresponding
	# locations
    cv2.imwrite('re.png',frame)
    return locs        

def predict1(frame):
    
    prototxtPath =("deploy.prototxt")
    weightsPath ="res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(2)
    boxes1 = detect_face(rgb,faceNet)
    print(boxes1)
    print('====')
    encodings = face_recognition.face_encodings(rgb, boxes1)
    names = []
    
    # loop over the facial embeddings
    for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
                    encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
            names.append(name)

    #cv2.imwrite('result.jpg',frame)
    print('done')
    #sys.exit()
    print(names)
    return names,frame






def predict(path):
    
    frame=cv2.imread(path)
    # frame=cv2.resize(frame,(300,300))
    prototxtPath =("deploy.prototxt")
    weightsPath ="res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = frame.shape[1] / float(rgb.shape[1])
    r=1
    print(2)
    boxes1 = detect_face(rgb,faceNet)

    # print(type(boxes[0]))
    print(boxes1)
    print('====')
    # s
    encodings = face_recognition.face_encodings(rgb, boxes1)
    names = []
    for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],
                    encoding)
            name = "Unknown"
            if True in matches:
                    
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
            names.append(name)
   

    cv2.imwrite('result.jpg',frame)
    print('done')
    print(names)
    sys.exit()
    
        


if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path1=askopenfilename()
    # dataset_masked\anushka\00000009_N95.jpg
    predict(path1)
    #dataset_masked/anushka/00000009.jpg
    #dataset_masked/trump/00000000_N95.jpg
