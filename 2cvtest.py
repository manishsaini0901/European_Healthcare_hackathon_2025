import cv2
import numpy as np

# Load the image
image = cv2.imread('compressed.png')  # Replace with your image path

# Convert to HSV color space for better color separation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color range for red-brownish cells in HSV space
lower_bound = np.array([0, 50, 50])  # Adjust these values for the exact color range
upper_bound = np.array([15, 255, 255])

# Apply color threshold to isolate the tumor cells based on color
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Bitwise the mask with the original image to get the filtered result
result = cv2.bitwise_and(image, image, mask=mask)

# Apply Gaussian blur to reduce noise and improve contour detection
blurred_image = cv2.GaussianBlur(result, (5, 5), 0)

# Convert to grayscale for edge detection
gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

# Use Canny edge detection to highlight boundaries
edges = cv2.Canny(gray, threshold1=50, threshold2=150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours that might just be noise, but keep all other shapes and sizes
valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 50]  # Reduced threshold to detect more cells

# Draw all contours on the original image
output_image = image.copy()

# Draw all valid contours with different colors (randomly)
for contour in valid_contours:
    cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)  # Green color for all contours

# Count the number of detected tumor cells (valid contours)
cell_count = len(valid_contours)

# Print the number of cells detected
print(f"Detected tumor cells: {cell_count}")

# Annotate the image with the number of detected cells
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(output_image, f'Cells Detected: {cell_count}', (10, 30), font, 1, (0, 0, 255), 2)

# Save the result image with all contours and count annotation
cv2.imwrite('output_image_with_all_cells_detected.jpg', output_image)

# No need to show the image, as per your request
