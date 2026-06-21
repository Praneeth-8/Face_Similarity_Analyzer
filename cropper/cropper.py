import cv2 

import mediapipe as mp



mp_face_detection = mp.solutions.face_detection
#load images
def crop_face(img_path):

    img = cv2.imread(img_path)

    with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.65) as face_detection:

        H1, W1, _ = img.shape
        img1_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        out = face_detection.process(img1_rgb)


        if not out.detections :
            print("Face not detected")
            exit()
        
        bbox1 = out.detections[0].location_data.relative_bounding_box
    
        
        x1 = int(bbox1.xmin * W1)
        y1 = int(bbox1.ymin * H1)
        w1 = int(bbox1.width * W1)
        h1 = int(bbox1.height * H1)

    # crop faces 
        face = img[y1-50:y1+h1+50, x1-50:x1+w1+50]

        cv2.imwrite("/home/praneeth/Project/similarity/data/face1.jpg", face)
face = "data/face2.jpeg"

crop_face(face)