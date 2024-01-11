import cv2
from matplotlib import pyplot as plt

# Opening image
img = cv2.imread("cat_face.jpg")

# OpenCV opens images as BGR
# we want it as RGB, also need a grayscale version
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Loading a pre-trained cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect objects
faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with rectangles around detected faces
plt.imshow(img_rgb)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()
