from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt


imgpath="2.jpg"
rf=Roboflow(api_key="42nNqT6OxhEeuC5TZS36")
Project=rf.workspace().project("smart-waste-management-h5yif-mwcpw")
model=Project.version(1).model
res=model.predict(imgpath,confidence=40,overlap=30).json()
predictions=res["predictions"]
Unique_classes=set(pred['class'] for pred in predictions)
print("Unique_classes:",Unique_classes)

for cls in Unique_classes:
    print(f"-{cls}")

xyxy=[]
confidence=[]
labels=[]
class_ids=[]


for pred in predictions:
    x1=int(pred["x"]-pred['width']/2)
    y1=int(pred["y"]-pred['height']/2)
    x2=int(pred["x"]-pred['width']/2)
    y2=int(pred["y"]-pred['height']/2)

    xyxy.append([x1,y1,x2,y2])
    confidence.append(pred['confidence'])
    class_ids.append(pred['class_ids'])
    labels.append(pred['class'])


detections=sv.Detections(
    xyxy=np.array(xyxy),
    confidence=np.array(confidence),
    class_ids=np.array(class_ids)
)

image=cv2.imread(imgpath)
image=cv2.imread(imgpath)
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

dox_annotator=sv.LabelAnnotator()
label_annotator=sv.LabelAnnotator()
annonated_image=dox_annotator.annotate(scene=image_rgb_copy(),detections=detections)
annonated_image=label_annotator.annotate(scene=annonated_image,detections=detections,labels=labels)


plt.figure(figsize=(10,10))
plt.imshow(annonated_image)
plt.axis("off")
plt.title("Annonated Image")
plt.show()
cv2.imwrite("output.jpg",cv2.cvtColor(annonated_image,cv2.COLOR_RGB2BGR))
print("Annonated image saved as output.jpg")


