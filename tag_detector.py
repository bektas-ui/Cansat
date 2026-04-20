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

# Manual calibration data from known distances and actual measurements
# Format: (pixel_size, distance_m, tag_size_m)
# All calibration data extracted from reference images
calibration_data = [
    (267.1, 6.00, 0.168),       # 6.00m: 267.1px, 0.168m tag
    (242.0, 6.50, 0.168),       # 6.50m: 242.0px, 0.168m tag
    (224.0, 7.00, 0.168),       # 7.00m: 224.0px, 0.168m tag
    (469.8, 80.10, 4.0),        # 80.10m: 469.8px, 4.0m tag (Final Apriltag 1.jpg)
]

# Calculate calibration constant: focal_length = (pixel_size * distance) / tag_size
focal_lengths = np.array([(d[0] * d[1]) / d[2] for d in calibration_data])
focal_length = np.mean(focal_lengths)

print(f"Calibration using multi-distance, multi-size reference data:")
print(f"Focal length constant: {focal_length:.1f}")
print(f"Standard deviation: {np.std(focal_lengths):.1f}")
print(f"Coefficient of variation: {np.std(focal_lengths) / focal_length * 100:.2f}%\n")

print("Calibration data:")
for px, dist, tag_size in calibration_data:
    fl = (px * dist) / tag_size
    print(f"  {dist:.2f}m: {px:.1f}px, {tag_size:.3f}m tag => fl = {fl:.1f}")

print("\nCalibration verification:")
for px, actual_dist, tag_size in calibration_data:
    calc_dist = (focal_length * tag_size) / px
    error_m = calc_dist - actual_dist
    error_pct = (error_m / actual_dist) * 100
    print(f"  {actual_dist:.2f}m: {calc_dist:.2f}m (error: {error_m:+.3f}m, {error_pct:+.2f}%)")

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

    unique_results = []
    seen_ids = set()
    for r in results:
        if r.tag_id in seen_ids:
            print(f"    Skipping duplicate detection of ID {r.tag_id}")
            continue
        seen_ids.add(r.tag_id)
        unique_results.append(r)

    print(f"  Using {len(unique_results)} unique tag(s) after deduplication")

    # Tag size maps (meters)
    tag_size_map = {
        'Final Apriltag 1.jpg': 4.0,  # 4x4 meter tag by filename
    }
    tag_size_by_id = {
        317: 1.0,  # AprilTag ID 317 is a 1 meter tag
    }
    
    # Default tag size for this image when no tag ID override matches
    default_tag_size_m = tag_size_map.get(img_path.name, 4.0)

    for r in unique_results:
        corners = np.array(r.corners, dtype=float)
        tag_size_m = tag_size_by_id.get(r.tag_id, default_tag_size_m)
        print(f"    Tag ID {r.tag_id} size: {tag_size_m:.2f}m")
        
        # Calculate all 4 sides for angle detection
        side1 = np.linalg.norm(corners[1] - corners[0])
        side2 = np.linalg.norm(corners[2] - corners[1])
        side3 = np.linalg.norm(corners[3] - corners[2])
        side4 = np.linalg.norm(corners[0] - corners[3])
        
        sides = np.array([side1, side2, side3, side4])
        
        # Get max side for distance calculation
        max_side = np.max(sides)
        
        # Calculate distance to tag using calibration formula
        # distance = (focal_length * tag_size) / pixel_size
        distance_m = (focal_length * tag_size_m) / max_side if max_side > 0 else 0
        
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
        print(f"        - 3D world distance: {actual_distance_m:.3f}m")
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
    
    # Add detailed overlays to the full-resolution image
    full_display_img = img_color.copy()
    
    # Draw detailed overlays on full-resolution image
    for r in unique_results:
        corners = np.array(r.corners, dtype=float)
        tag_size_m = tag_size_by_id.get(r.tag_id, default_tag_size_m)
        
        # Calculate all 4 sides for angle detection
        side1 = np.linalg.norm(corners[1] - corners[0])
        side2 = np.linalg.norm(corners[2] - corners[1])
        side3 = np.linalg.norm(corners[3] - corners[2])
        side4 = np.linalg.norm(corners[0] - corners[3])
        max_side = np.max([side1, side2, side3, side4])
        
        # Calculate distance using the measured side length and focal_length calibration
        distance_m = (focal_length * tag_size_m) / max_side if max_side > 0 else 0
        
        # Use original coordinates (no scaling needed for full resolution)
        tag_center = np.mean(corners, axis=0)
        camera_center = np.array([cx_actual, cy_actual])
        
        # Draw tag corners on full-resolution image
        corners_int = corners.astype(int)
        cv2.polylines(full_display_img, [corners_int], True, (0, 255, 0), 3)  # Thicker lines for visibility
        
        # Draw tag center
        center_int = tag_center.astype(int)
        cv2.circle(full_display_img, tuple(center_int), 8, (0, 0, 255), -1)  # Larger circle
        cv2.putText(full_display_img, f"ID: {r.tag_id}", tuple(center_int + np.array([15, -15])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)  # Larger text
        
        # Draw camera center
        camera_center_int = camera_center.astype(int)
        cv2.circle(full_display_img, tuple(camera_center_int), 8, (255, 0, 0), -1)  # Larger circle
        cv2.putText(full_display_img, "Camera Center", tuple(camera_center_int + np.array([15, -15])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)  # Larger text
        
        # Draw line from camera center to tag center
        cv2.line(full_display_img, tuple(camera_center_int), tuple(center_int), (200, 200, 0), 3)  # Thicker line
        
        # Calculate distances
        dist_px, dist_3d_m, off_x_px, off_y_px, off_x_m, off_y_m = calculate_center_distance(
            tag_center, camera_center, cx_actual, cy_actual, fx, fy, distance_m, max_side=max_side
        )
        
        # Calculate actual distance using Pythagorean theorem
        offset_in_plane = np.sqrt(off_x_m**2 + off_y_m**2)
        actual_distance_m = np.sqrt(distance_m**2 - offset_in_plane**2) if distance_m**2 >= offset_in_plane**2 else 0
        
        # Draw distance info with larger, more visible text
        cv2.putText(full_display_img, f"DISTANCE: {distance_m:.2f}m", tuple(center_int + np.array([15, 25])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)  # Large red text for distance
        cv2.putText(full_display_img, f"Tag Size: {tag_size_m:.2f}m", tuple(center_int + np.array([15, 60])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(full_display_img, f"REAL ALTITUDE: {actual_distance_m:.2f}m", 
                   tuple(center_int + np.array([15, 90])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Green text for real altitude
    
    # Save the full-resolution image with detailed overlays
    full_display_path = output_folder / f"full_display_{img_path.name}"
    cv2.imwrite(str(full_display_path), full_display_img)
    print(f"  Saved full-resolution display to: {full_display_path}")
    
    # Also save the resized version for reference
    original_h, original_w = img_color.shape[:2]
    display_size = (1200, 900)
    scale_x = display_size[0] / original_w
    scale_y = display_size[1] / original_h
    
    display_img = cv2.resize(full_display_img, display_size)
    display_output_path = output_folder / f"display_{img_path.name}"
    cv2.imwrite(str(display_output_path), display_img)
    print(f"  Saved resized display to: {display_output_path}")