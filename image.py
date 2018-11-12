import cv2
import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.nan)
from helper import preprocess, centeredCrop



image = "Blender/test_data/data_24.png"
X = cv2.imread(image)
print(X.shape)

X = cv2.resize(X, (455, 256))
X = centeredCrop(X, 224)



img = Image.fromarray(X, 'RGB')
img.save('my.png')
img.show()

new_x = preprocess([image])
print(new_x[0][0].shape)
img2 = Image.fromarray(new_x[0][0], 'RGB')
img2.save('my2.png')
img2.show()