import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import InceptionResnetV1

#Load model
model = InceptionResnetV1(pretrained="vggface2").eval()

#preprocess from cropper
model = InceptionResnetV1(pretrained="vggface2").eval()

def get_embedding(face_crop):
    # BGR -> RGB
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # NumPy -> PIL
    face = Image.fromarray(face_crop)

    # Resize
    face = face.resize((160, 160))

    # PIL -> NumPy
    face = np.array(face).astype(np.float32)

    # Normalize
    face = face / 255.0

    # HWC -> CHW
    face = np.transpose(face, (2, 0, 1))

    # NumPy -> Tensor
    face = torch.tensor(face, dtype=torch.float32)

    # Add batch dimension
    face = face.unsqueeze(0)

    # Get embedding
    with torch.no_grad():
        embedding = model(face)

    return embedding.squeeze().numpy()