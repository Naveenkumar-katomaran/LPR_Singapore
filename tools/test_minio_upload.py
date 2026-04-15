import json
import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.minio_utils import MinioClient

def test_minio():
    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.json'))
    with open(config_path) as f:
        config = json.load(f)

    if "minio" not in config:
        print("Error: 'minio' configuration not found in config.json")
        return

    print("Testing MinIO connection...")
    try:
        client = MinioClient(config["minio"])
        print("MinIO Client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize MinIO Client: {e}")
        return

    # Create a dummy image
    print("Creating dummy image...")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, encoded = cv2.imencode('.jpg', img)
    data = encoded.tobytes()

    # Upload test
    print("Uploading dummy image...")
    test_key = "tests/test_upload.jpg"
    try:
        result = client.upload_bytes(data, test_key)
        if result:
            print(f"Successfully uploaded to {test_key}")
            # Generate URL
            url = client.get_url(test_key)
            public_url = client.get_public_url(test_key)
            print(f"Presigned URL: {url}")
            print(f"Public/Direct URL: {public_url}")
        else:
            print("Upload failed.")
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == "__main__":
    test_minio()
