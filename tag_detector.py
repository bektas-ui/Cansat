# ...existing code...
import cv2
import numpy as np
import os
from pathlib import Path
from math import atan2, asin, degrees, sqrt, atan
# import apriltag
from pupil_apriltags import Detector as AprilTagDetector

# Initialize detector
detector = AprilTagDetector(families='tag36h11')


def calculate_center_distance(tag_center_px, camera_center_px, cx, cy, fx, fy, distance_m, max_side=None):
    """
    Calculate the distance from camera center to tag center in both 2D (pixels) and 3D (meters).
    
    Args:
        tag_center_px: tuple (x, y) tag center in pixel coordinates
        camera_center_px: tuple (x, y) camera center in pixel coordinates (typically cx, cy)
        cx, cy: camera principal point in pixels
        fx, fy: camera focal lengths in pixels
        distance_m: distance from camera to tag plane in meters
        max_side: maximum side length in pixels (for center distance calculation)
    
    Returns:
        distance_px: distance in pixels
        distance_3d_m: 3D distance from camera center to tag center in meters
        offset_x_px: horizontal offset in pixels
        offset_y_px: vertical offset in pixels
        offset_x_m: horizontal offset in meters (at tag distance)
        offset_y_m: vertical offset in meters (at tag distance)
    """
    
    # Pixel distance from camera center
    offset_x_px = tag_center_px[0] - camera_center_px[0]
    offset_y_px = tag_center_px[1] - camera_center_px[1]
    
    distance_px = np.sqrt(offset_x_px**2 + offset_y_px**2)
    
    # Calculate angular offsets from pixel coordinates
    angle_x = np.arctan2(offset_x_px, fx)  # horizontal angle
    angle_y = np.arctan2(offset_y_px, fy)  # vertical angle
    
    # Convert angular offset to 3D world coordinates using the distance to the tag plane
    # offset = distance_m * tan(angle)
    offset_x_m = distance_m * np.tan(angle_x)
    offset_y_m = distance_m * np.tan(angle_y)
    
    # Calculate 3D distance from camera center to tag center using Pythagorean theorem
    # distance_to_plane is the hypotenuse
    # distance_to_plane^2 = distance_to_center^2 + offset_in_plane^2
    # therefore: distance_to_center = sqrt(distance_to_plane^2 - offset_in_plane^2)
    offset_in_plane = np.sqrt(offset_x_m**2 + offset_y_m**2)
    distance_3d_m = np.sqrt(distance_m**2 - offset_in_plane**2) if distance_m**2 >= offset_in_plane**2 else 0
    
    return distance_px, distance_3d_m, offset_x_px, offset_y_px, offset_x_m, offset_y_m



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
    (78.4, 2.00),    # BOMA1.jpg: 78.4px at 2.00m (straight-on, accurate)
    (40.1, 3.80),    # gab380cm.jpg: 40.1px at 3.80m
    (34.0, 4.50),    # gab450cm.jpg: 34.0px at 4.50m
]

# Create calibration curve using the actual measurement data
pixel_sizes = np.array([d[0] for d in calibration_data])
distances = np.array([d[1] for d in calibration_data])

# Calculate calibration constant: k = pixel_size * distance
k_values = pixel_sizes * distances
k = np.mean(k_values)

print(f"Calibration using extracted reference data:")
print(f"  0.80m: 195.0px => k = {195.0 * 0.80:.1f}")
print(f"  1.30m: 118.9px => k = {118.9 * 1.30:.1f}")
print(f"  1.80m:  85.5px => k = {85.5 * 1.80:.1f}")
print(f"  2.00m:  78.4px => k = {78.4 * 2.00:.1f} (BOMA1 - straight-on)")
print(f"  3.80m:  40.1px => k = {40.1 * 3.80:.1f}")
print(f"  4.50m:  34.0px => k = {34.0 * 4.50:.1f}")
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
    
    # Set camera center to image center
    img_h, img_w = img.shape
    cx_actual = img_w / 2.0
    cy_actual = img_h / 2.0
    
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
        
        # Nominal tag size
        tag_size_cm = 16.8  # AprilTag side in cm
        
        # Calculate distance to tag using calibration constant
        # distance_m = k / pixel_size
        distance_m = k / max_side if max_side > 0 else 0
        
        # Calculate distance from camera center to tag center
        tag_center = np.mean(corners, axis=0)
        camera_center = np.array([cx_actual, cy_actual])
        dist_px, dist_3d_m, off_x_px, off_y_px, off_x_m, off_y_m = calculate_center_distance(
            tag_center, camera_center, cx_actual, cy_actual, fx, fy, distance_m, max_side=max_side
        )
        
        # Calculate actual distance using Pythagorean theorem
        # distance_to_plane is the hypotenuse
        offset_in_plane = np.sqrt(off_x_m**2 + off_y_m**2)
        actual_distance_m = np.sqrt(distance_m**2 - offset_in_plane**2) if distance_m**2 >= offset_in_plane**2 else 0
        
        print(f"    Tag ID: {r.tag_id}")
        print(f"      Distance to tag plane: {distance_m:.3f}m")
        print(f"      Distance (camera center to tag center):")
        print(f"        - 2D pixel distance: {dist_px:.1f}px")
        print(f"        - 3D world distance (Pythagorean): {actual_distance_m:.3f}m")
        print(f"        - Horizontal offset: {off_x_px:.1f}px ({off_x_m:.3f}m)")
        print(f"        - Vertical offset: {off_y_px:.1f}px ({off_y_m:.3f}m)")
        print(f"      Pixel sizes: Top={side1:.1f}, Right={side2:.1f}, Bottom={side3:.1f}, Left={side4:.1f}px")
        
        # Draw tag corners on image
        corners_int = corners.astype(int)
        cv2.polylines(img_color, [corners_int], True, (0, 255, 0), 2)
        
        # Draw tag center and ID
        center = np.mean(corners, axis=0).astype(int)
        cv2.circle(img_color, tuple(center), 5, (0, 0, 255), -1)
        cv2.putText(img_color, f"ID: {r.tag_id}", tuple(center + np.array([10, -10])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Store tag data for later display rendering (we'll draw on resized image)
    
    # Save the processed image
    output_path = output_folder / f"detected_{img_path.name}"
    cv2.imwrite(str(output_path), img_color)
    print(f"  Saved to: {output_path}")
    
    # Display the image with overlays (resized for better viewing) with properly scaled coordinates
    original_h, original_w = img_color.shape[:2]
    display_size = (300, 225)
    scale_x = display_size[0] / original_w
    scale_y = display_size[1] / original_h
    
    display_img = cv2.resize(img_color, display_size)
    
    # Redraw everything with scaled coordinates on the resized image
    for r in results:
        corners = np.array(r.corners, dtype=float)
        
        # Calculate all 4 sides for angle detection
        side1 = np.linalg.norm(corners[1] - corners[0])
        side2 = np.linalg.norm(corners[2] - corners[1])
        side3 = np.linalg.norm(corners[3] - corners[2])
        side4 = np.linalg.norm(corners[0] - corners[3])
        max_side = np.max([side1, side2, side3, side4])
        
        # Calculate distance using the measured side length and calibration constant
        tag_size_cm = 16.8  # AprilTag side in cm
        distance_m = k / max_side if max_side > 0 else 0
        
        # Scale coordinates
        corners_scaled = corners * np.array([scale_x, scale_y])
        tag_center = np.mean(corners, axis=0)
        center_scaled = tag_center * np.array([scale_x, scale_y])
        camera_center_scaled = np.array([cx_actual, cy_actual]) * np.array([scale_x, scale_y])
        
        # Draw tag corners on resized image
        corners_int = corners_scaled.astype(int)
        cv2.polylines(display_img, [corners_int], True, (0, 255, 0), 2)
        
        # Draw tag center
        center_int = center_scaled.astype(int)
        cv2.circle(display_img, tuple(center_int), 5, (0, 0, 255), -1)
        cv2.putText(display_img, f"ID: {r.tag_id}", tuple(center_int + np.array([10, -10])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw camera center (now correctly positioned on resized image)
        camera_center_int = camera_center_scaled.astype(int)
        cv2.circle(display_img, tuple(camera_center_int), 5, (255, 0, 0), -1)
        cv2.putText(display_img, "Camera Center", tuple(camera_center_int + np.array([10, -10])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw line from camera center to tag center
        cv2.line(display_img, tuple(camera_center_int), tuple(center_int), (200, 200, 0), 2)
        
        # Calculate distances
        camera_center_orig = np.array([cx_actual, cy_actual])
        dist_px, dist_3d_m, off_x_px, off_y_px, off_x_m, off_y_m = calculate_center_distance(
            tag_center, camera_center_orig, cx_actual, cy_actual, fx, fy, distance_m, max_side=max_side
        )
        
        # Calculate actual distance using Pythagorean theorem
        actual_distance_m = np.sqrt(distance_m**2 - offset_in_plane**2) if distance_m**2 >= offset_in_plane**2 else 0
        
        # Draw distance info
        cv2.putText(display_img, f"Dist (plane): {distance_m:.2f}m", tuple(center_int + np.array([10, 15])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(display_img, f"Size: {max_side:.0f}px (16.8cm)", tuple(center_int + np.array([10, 35])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_img, f"Center dist (Pythagoras): {actual_distance_m:.3f}m ({dist_px:.0f}px)", 
                   tuple(center_int + np.array([10, 55])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    
    cv2.imshow(f"Tags: {img_path.name}", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()