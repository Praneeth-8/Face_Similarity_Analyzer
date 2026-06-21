import cv2
import numpy as np
from cropper import crop_face
from embed import get_embedding

img1_path = "data/face1.jpg" #use the correspoinding path of images
img2_path = "data/face2.jpg"

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise ValueError("Failed to load one or both images.")

# # Crop faces

face1 = img1
face2 = img2

if face1 is None or face2 is None:
    raise ValueError("Face not detected in one or both images.")

emb1 = get_embedding(face1)
emb2 = get_embedding(face2)

similarity = np.dot(emb1,emb2) / (
    np.linalg.norm(emb1) * np. linalg.norm(emb2)
)

print(f"Similarity Score: {similarity:.4f}")