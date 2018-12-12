import cv2
import numpy as np
import os


##################### KNN CODE ########################
def dist(x1,x2):
    return np.sqrt(np.sum((x2-x1)**2))


def knn(x_train,y_train,q_point,k=11):
    dist_val = []
    
    m = x_train.shape[0]
    for ix in range(m):
        tempdist = dist(q_point,x_train[ix,:])
        dist_val.append((tempdist,y_train[ix]))
        
    dist_val = sorted(dist_val)
    dist_val = dist_val[:k]
#     print(dist_val)
    
    y = np.array(dist_val)
#     print(y.shape)
    te = np.unique(y[:,1],return_counts=True)
#     print(te)
    idx = te[1].argmax()
#     print(idx)
    prediction = te[0][idx]
    return prediction
#####################################################



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

skip = 0
face_data = []
dataset_path = './data/'

labels = []

class_id = 0
name = {}

for fx in os.listdir(dataset_path):

	if(fx.endswith('.npy')):
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		target = class_id* np.ones((data_item.shape[0]))
		name[class_id] = fx[:-4]
		class_id+=1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0)

print(face_dataset.shape)
print(face_labels.shape)

while True:

	ret, frame = cap.read()

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face

		offset = 10

		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]

		face_section = cv2.resize(face_section,(100,100))

		out = knn(face_dataset,face_labels,face_section.flatten())

		predicted_name = name[out]
		cv2.putText(frame,predicted_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,30,50),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(30,230,200),2)

	cv2.imshow('Faces',frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

