import numpy as np
import torch
import cv2 as cv

#load model
model = torch.load('best.pt')

#load picture
Camera = cv.VideoCapture(0)
while True:
    ret,frame = Camera.read()
#detect
    detections = model(frame)
    print(detections)
    #print results
    results = detections.pandas().xyxy[0].to_dict(orient="records")
    x= np.array(results)
#filter
    for result in results:
        confidence = result['confidence']
        confidence = round(confidence,2)      
        confidence = str(confidence)
        name = result['name']
        print(type(name))
        clas = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        frame = cv.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        a = name + "  " + confidence
        cv.putText(frame, a,(x1,y1), cv.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 2, cv.LINE_AA)
        # print (name,x1,y1,x2,x2)
        cv.imshow("Hello",frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
Camera.release()
cv.destroyAllWindows()
