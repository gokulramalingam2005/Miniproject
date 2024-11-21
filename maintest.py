import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread(r'C:\Users\abhis\OneDrive\Desktop\Brain tumor\pred\pred13.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)
threshold = 0.5

# Convert probability values to binary decision
binary_prediction = 1 if result[0, 1] > threshold else 0

print(binary_prediction)
if(binary_prediction==1):
    {
        print("THE MRI SCAN IS DETECTED WITH TUMOR\nHope your recovery is short,sweet and strong.")
    }
else:
    {
        print("THIS MRI SCAN DOES NOT HAVE TUMOR\nGOOD TO GO :)")  
    }
