import cv2
import numpy as np
from pathlib import Path
from pupil_apriltags import Detector as AprilTagDetector

# Initialize detector
detector = AprilTagDetector(families='tag36h11')

img_path = Path('img') / 'BOMA.jpg'
img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
results = detector.detect(img)

if results:
    r = results[0]
    print(f"Tag ID: {r.tag_id}")
    print(f"Has pose: {r.pose_t is not None}")
    if r.pose_t is not None:
        print(f"Pose translation: {r.pose_t}")
        print(f"Pose rotation: {r.pose_R}")
    print(f"Has pose_err: {r.pose_err is not None}")
    if r.pose_err is not None:
        print(f"Pose error: {r.pose_err}")
