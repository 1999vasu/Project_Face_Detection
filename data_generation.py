import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)

skip = 0
face_data = []
dataset_path = './data/'

file_name = input('Enter person name:')

while(True):
	ret,frame = cap.read()

	if(ret == False):
		continue
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray,1.3,5)

	k=1

	faces = sorted(faces, key = lambda x: x[2]*x[3],reverse = True)

	skip+=1

	for face in faces[:1]:
		x,y,w,h = face

		offsides =10
		face_section = frame[y-offsides:y+h+offsides,x-offsides:x+w+offsides]

		face_section = cv2.resize(face_section,(100,100))

		if(skip%10==0):

			face_data.append(face_section)
			print(len(face_data))

		cv2.imshow(str(k), face_section)
		k += 1

		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Convert face list to numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# Save the dataset in filesystem
np.save(dataset_path + file_name, face_data)
print("Dataset saved at: {}".format(dataset_path + file_name + '.npy'))

cap.release()
cv2.destroyAllWindows()






