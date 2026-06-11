# pipeline.py
import cv2
from utils import get_device
from embedder import FaceEmbedder
from verify import verify_embeddings


def capture_selfie() -> str:
    """
    Capture a selfie and save it as temp_selfie.jpg
    Returns:
        str: path of saved selfie image
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot access camera!")

    print("capture selfie")

    img = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Selfie", frame)

        key = cv2.waitKey(1)
        if key == 32:    # SPACE
            img = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    # save selfie
    save_path = "temp_selfie.jpg"
    cv2.imwrite(save_path, img)
    print(f" Selfie saved → {save_path}")
    return save_path



def verify_user(id_image_path: str, threshold: float = 0.5):
    """
    Full pipeline:
      - extract embedding from ID
      - capture selfie
      - extract embedding from selfie
      - compare
    """

    device = get_device()
    print(f"Using device: {device}")

    embedder = FaceEmbedder(device)

    print("\nExtracting face from ID...")
    id_emb = embedder.extract(id_image_path)
    if id_emb is None:
        print("No face detected in ID!")
        return False

    print("ID face embedding extracted")

    print("\nCapturing selfie...")
    selfie_path = capture_selfie()

    print("\nExtracting face from selfie...")
    selfie_emb = embedder.extract(selfie_path)
    if selfie_emb is None:
        print("No face detected in selfie!")
        return False

    print("Selfie face embedding extracted")

    print("\nComparing embeddings...")
    is_match, dist = verify_embeddings(id_emb, selfie_emb, threshold)

    print(f"Distance = {dist:.4f}")
    if is_match:
        print(" SAME PERSON ")
    else:
        print("DIFFERENT PERSON ")

    return is_match
