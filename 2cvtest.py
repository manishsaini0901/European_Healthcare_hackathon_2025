import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os

print("Starting cell detection...")

# Check if file exists
print("Checking file path...")
if not os.path.exists('compressed.png'):
    print("Error: File not found.")
    exit()
print("File found.")

# Load the image
image = cv2.imread('compressed.png')
if image is None:
    print("Error: Could not load image. Please check the file integrity.")
    exit()
print("Image loaded successfully.")

# Convert to HSV color space
print("Converting to HSV...")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print("Converted to HSV.")

# Reshape pixels for clustering
print("Reshaping image for clustering...")
pixels = hsv_image.reshape((-1, 3))
print(f"Pixels reshaped: {pixels.shape}")

# Fit MiniBatchKMeans clustering to handle large datasets
print("Fitting MiniBatchKMeans clustering...")
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=5, max_iter=50, batch_size=10000)
kmeans.fit(pixels)
print("MiniBatchKMeans fitted.")

# Get the predicted cluster for each pixel
print("Predicting clusters...")
clusters = kmeans.predict(pixels).reshape(hsv_image.shape[:2])
print(f"Clusters predicted: {np.unique(clusters)}")

# Get the cluster centers (mean colors of clusters)
print("Extracting cluster centers...")
cluster_centers = kmeans.cluster_centers_
print("Cluster centers:", cluster_centers)

# Automatically assign clusters based on HSV brightness (V channel)
print("Sorting clusters by brightness...")
sorted_indices = np.argsort(cluster_centers[:, 2])
print(f"Sorted cluster indices: {sorted_indices}")

# Assign the categories based on brightness
light_cluster = sorted_indices[2]
medium_cluster = sorted_indices[1]
dark_cluster = sorted_indices[0]
print(f"Assigned clusters -> Light: {light_cluster}, Medium: {medium_cluster}, Dark: {dark_cluster}")

# Create separate masks for each cluster type
print("Creating masks...")
mask_light = (clusters == light_cluster).astype(np.uint8) * 255
mask_medium = (clusters == medium_cluster).astype(np.uint8) * 255
mask_dark = (clusters == dark_cluster).astype(np.uint8) * 255
print("Masks created.")

# Apply morphological operations to clean up the masks
print("Applying morphological operations...")
kernel = np.ones((2, 2), np.uint8)
mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
mask_medium = cv2.morphologyEx(mask_medium, cv2.MORPH_CLOSE, kernel)
mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)
print("Morphological operations applied.")

# Find contours for each type
print("Finding contours...")
contours_light, _ = cv2.findContours(mask_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_medium, _ = cv2.findContours(mask_medium, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_dark, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours found -> Light: {len(contours_light)}, Medium: {len(contours_medium)}, Dark: {len(contours_dark)}")

# Calculate average size of dark (tumor) cells
print("Calculating average size of tumor cells...")
if len(contours_dark) > 0:
    avg_dark_size = np.mean([cv2.contourArea(c) for c in contours_dark])
else:
    avg_dark_size = 0
print(f"Average tumor cell size: {avg_dark_size}")

# Create a copy of the original image to draw contours
print("Creating output image...")
output_image = image.copy()

# Draw contours for each type
print("Drawing contours...")

# Light (normal) cells
for contour in contours_light:
    if cv2.contourArea(contour) > 3:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 1:
            cv2.circle(output_image, (int(x), int(y)), int(radius), (0, 255, 0), 1)  # Green for normal cells
print("Drawn normal cell contours.")

# Medium cells
for contour in contours_medium:
    if cv2.contourArea(contour) > 3:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 1:
            cv2.circle(output_image, (int(x), int(y)), int(radius), (255, 255, 0), 1)  # Yellow for intermediate cells
print("Drawn intermediate cell contours.")

# Tumor cells (only smaller than average size)
for contour in contours_dark:
    if cv2.contourArea(contour) > 3 and cv2.contourArea(contour) < avg_dark_size:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 1:
            cv2.circle(output_image, (int(x), int(y)), int(radius), (0, 0, 255), 1)  # Red for tumor cells
print("Drawn tumor cell contours.")

# Count the number of each type
print("Counting cells...")
count_light = len(contours_light)
count_medium = len(contours_medium)
count_dark = sum(1 for c in contours_dark if cv2.contourArea(c) < avg_dark_size)
print(f"Counts -> Normal: {count_light}, Intermediate: {count_medium}, Tumor: {count_dark}")

# Annotate the image with the counts
print("Annotating image...")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(output_image, f'Normal: {count_light}', (10, 30), font, 0.5, (0, 255, 0), 1)
cv2.putText(output_image, f'Intermediate: {count_medium}', (10, 50), font, 0.5, (255, 255, 0), 1)
cv2.putText(output_image, f'Tumor: {count_dark}', (10, 70), font, 0.5, (0, 0, 255), 1)

# Identify and mark the densest tumor area
print("Identifying densest tumor area...")
if len(contours_dark) > 0:
    max_cluster = max(contours_dark, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cluster)
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Magenta for densest tumor area
print("Densest tumor area marked.")

# Save the result image
cv2.imwrite('output_image_with_manual_thresholds.jpg', output_image)
print("Result saved as 'output_image_with_manual_thresholds.jpg'")

print("Cell detection complete.")
