## **Pothole Detection**

### 🎯 **Goal**

To do Pothole Detection Using YOLO

### 🧵 **Dataset**

Dataset was made by me and uploaded in Roboflow to annotate
[Dataset](https://universe.roboflow.com/nit-raipur-nz8f9/pothole-2-top-view/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

### 🧾 **Description**

I used YOLOv8 to do Pothole Detection which had only 1 class

### 🧮 **What I had done!**

1. First we need to import ultralytics. 
2. Import the dataset from Roboflow(Dataset can be imported from any source but it just needs to be anotated in YOLOv8 format).
3. Download the yolov8s.pt file from official Github of YOLO and use it to train the model.
4. You can extract the best.pt file generated from training and use the code given below for realtime testing with WebCamera.
5. Use the data.yaml file to get the proper class names.

```python
from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture(0)  # For Video

model = YOLO("best.pt")


classNames = ['PotHoles']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.50:
                if currentClass =='PotHoles' :
                    myColor = (0, 255, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=2,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=3)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

```

### 🚀 **Models Implemented**

YOLOv8 model has been used here because it is very fast to do Real-time Object Detection when compared to ANN or CNN. It's result might not be good locally but when considered globally it provides quite impressive results.

### 📚 **Libraries Needed**

1. Ultralytics 
2. YOLOv8.2.18 🚀 
3. Python-3.10.12 
4. torch-2.2.1+cu121

### 📊 **Exploratory Data Analysis Results**

![Confusion Matrix](results\confusionMatrix.png)
![Results](results/results.png)
![Train Batch](results/trainbatch.jpeg)
![Validation Batch](results/validation.jpeg)

### 📈 **Performance of the Models based on the Accuracy Scores**

# Model Summary

- **Layers**: 168
- **Parameters**: 11,129,067
- **Gradients**: 0
- **GFLOPs**: 28.5

## Performance Metrics

| Class           | Images | Instances | Box(P) | R    | mAP50 | mAP50-95 |
|:----------------|:-------|:----------|:------|:-----|:------|:---------|
| **all**         | 30     | 102        | 0.952 | 0.375    | 0.428 | 0.296    |
| **Pothole**| 29    | 101         | 0.904     | 0.749    | 0.856 | 0.593    |





### 📢 **Conclusion**

YOLOv8 perform remarkably for Pothole Detection giving and accuracy of almost 95% with such varied form of data and classes.
### ✒️ **Your Signature**

**Name- Utsab Samadder**
**Email-utsab.samadder@gmail.com**
**LinkedIn-https://www.linkedin.com/in/utsab-samadder/**
**Github-https://github.com/utsabsamadder**

