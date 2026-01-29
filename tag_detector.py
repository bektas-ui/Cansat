# ...existing code...
import cv2
import numpy as np
import os
from pathlib import Path
# import apriltag
from pupil_apriltags import Detector as AprilTagDetector

# Initialize detector
detector = AprilTagDetector(families='tag36h11')

# Get all image files from img folder
img_folder = Path('img')
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
image_files = [f for f in img_folder.glob('*') if f.suffix.lower() in image_extensions]

print(f"Found {len(image_files)} image(s) to process")

# camera intrinsics (fx, fy, cx, cy) from calibration
# Will be calibrated using reference images
fx, fy, cx, cy = 1000, 1000, 320, 240  # Initial placeholder values
tag_size_m = 0.168  # real tag side length in meters

# Manual calibration data from known distances and actual measurements
# All calibration data extracted from reference images
calibration_data = [
    (195.0, 0.80),   # gab80cm.jpg: 195.0px at 0.80m
    (118.9, 1.30),   # gab130cm.jpg: 118.9px at 1.30m
    (85.5, 1.80),    # gab180cm.jpg: 85.5px at 1.80m
    (78.7, 2.10),    # BOMA1.jpg: 78.7px at 2.10m (straight-on, accurate)
    (40.1, 3.80),    # gab380cm.jpg: 40.1px at 3.80m
    (34.0, 4.50),    # gab450cm.jpg: 34.0px at 4.50m
    (45.6, 3.40),    # bord1m.jpg: 45.6px at 3.40m (340cm)
]

# Create calibration curve using the actual measurement data
pixel_sizes = np.array([d[0] for d in calibration_data])
distances = np.array([d[1] for d in calibration_data])

# Calculate calibration constant: k = pixel_size * distance
k_values = pixel_sizes * distances
k = np.mean(k_values)

print(f"Calibration using extracted reference data:")
print(f"  0.80m: 195.0px → k = {195.0 * 0.80:.1f}")
print(f"  1.30m: 118.9px → k = {118.9 * 1.30:.1f}")
print(f"  1.80m:  85.5px → k = {85.5 * 1.80:.1f}")
print(f"  2.10m:  78.7px → k = {78.7 * 2.10:.1f} (BOMA1 - straight-on)")
print(f"  3.40m:  45.6px → k = {45.6 * 3.40:.1f}")
print(f"  3.80m:  40.1px → k = {40.1 * 3.80:.1f}")
print(f"  4.50m:  34.0px → k = {34.0 * 4.50:.1f}")
print(f"  Average calibration constant k: {k:.1f}")

objp = np.array([[-tag_size_m/2, -tag_size_m/2, 0],
                 [ tag_size_m/2, -tag_size_m/2, 0],
                 [ tag_size_m/2,  tag_size_m/2, 0],
                 [-tag_size_m/2,  tag_size_m/2, 0]], dtype=float)

# Process each image
print("\nProcessing all images...")

# Create output folder for results
output_folder = Path('output')
output_folder.mkdir(exist_ok=True)

for img_path in image_files:
    print(f"\nProcessing: {img_path}")
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to color for drawing
    
    if img is None:
        print(f"  Error: Could not read image {img_path}")
        continue
    
    results = detector.detect(img)
    print(f"  Detected {len(results)} tag(s)")
    
    for r in results:
        corners = np.array(r.corners, dtype=float)
        
        # Calculate all 4 sides for angle detection
        side1 = np.linalg.norm(corners[1] - corners[0])
        side2 = np.linalg.norm(corners[2] - corners[1])
        side3 = np.linalg.norm(corners[3] - corners[2])
        side4 = np.linalg.norm(corners[0] - corners[3])
        
        sides = np.array([side1, side2, side3, side4])
        
        # Get max side for distance calculation
        max_side = np.max(sides)
        min_side = np.min(sides)
        
        # Calculate distance using calibration constant
        # distance = k / pixel_size
        # Use max_side instead of average to avoid errors from angled tags
        distance_m = (k / max_side) if max_side > 0 else 0
        
        print(f"    Tag ID: {r.tag_id}, distance: {distance_m:.3f}m, pixel_size: {max_side:.1f}px")
        
        # Draw tag corners on image
        corners_int = corners.astype(int)
        cv2.polylines(img_color, [corners_int], True, (0, 255, 0), 2)
        
        # Draw tag center and ID
        center = np.mean(corners, axis=0).astype(int)
        cv2.circle(img_color, tuple(center), 5, (0, 0, 255), -1)
        cv2.putText(img_color, f"ID: {r.tag_id}", tuple(center + np.array([10, -10])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw distance info
        cv2.putText(img_color, f"Dist: {distance_m:.2f}m", tuple(center + np.array([10, 15])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw pixel size
        cv2.putText(img_color, f"Size: {max_side:.0f}px", tuple(center + np.array([10, 35])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Save the processed image
    output_path = output_folder / f"detected_{img_path.name}"
    cv2.imwrite(str(output_path), img_color)
    print(f"  Saved to: {output_path}")
    
    # Display the image with overlays (resized for better viewing)
    display_img = cv2.resize(img_color, (1200, 900))
    cv2.imshow(f"Tags: {img_path.name}", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()