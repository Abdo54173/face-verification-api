from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from typing import Optional
from utils import load_image, get_device


class FaceEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or get_device()

        # Face detector
        self.detector = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            post_process=True,
            device=self.device
        )

        # Face embedding model (pretrained)
        self.model = (
            InceptionResnetV1(pretrained='vggface2')
            .eval()
            .to(self.device)
        )

    def extract(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract a 512-dim facial embedding.
        Returns None if no face is detected.
        """
        img = load_image(image_path)
        face = self.detector(img)

        if face is None:
            print(f"No face detected in {image_path}")
            return None

        face = face.unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(face).cpu().numpy().squeeze(0)

        #  Normalize the embedding 
        emb = emb / np.linalg.norm(emb)
        return emb
