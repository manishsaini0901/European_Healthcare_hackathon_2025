import cv2
import numpy as np

# Load image
image = cv2.imread('compressed.png')

# Convert to HSV to target red hues
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define broader lower and upper bounds for red shades
lower_red1 = np.array([0, 50, 20])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 20])
upper_red2 = np.array([180, 255, 255])

# Create masks for red shades
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(mask, (7, 7), 0)

# Adaptive thresholding for varying shades
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 4)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Parameters
min_size = 15
max_size = 5000
min_circularity = 0.4

# Counters
low_intensity = 0
medium_intensity = 0
high_intensity = 0

for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if min_size < area < max_size and circularity > min_circularity:
        # Calculate intensity
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(hsv, mask=mask)[2]
        
        if mean_val < 100:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 1)  # Blue for low intensity
            low_intensity += 1
        elif 100 <= mean_val < 180:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 1)  # Green for medium intensity
            medium_intensity += 1
        else:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)  # Red for high intensity
            high_intensity += 1

# Display result
cv2.putText(image, f'Low: {low_intensity} Medium: {medium_intensity} High: {high_intensity}', 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow('Detected Cells', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite('/mnt/data/cell_detection_output.png', image)

print(f"Low intensity cells: {low_intensity}")
print(f"Medium intensity cells: {medium_intensity}")
print(f"High intensity cells: {high_intensity}")
