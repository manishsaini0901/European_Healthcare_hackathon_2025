import cv2
import numpy as np
from sklearn.cluster import KMeans

print("Starting cell detection...")

# Load the image
image = cv2.imread('compressed.png')  # Replace with your image path
print("Image loaded:", image is not None)

if image is None:
    print("Error: Could not load image. Please check the file path and integrity.")
    exit()

# Convert to HSV color space for better color separation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print("Converted to HSV.")

# Reshape pixels for clustering
pixels = hsv_image.reshape((-1, 3))

# Randomly sample 10% of the pixels for clustering to save memory
sampled_pixels = pixels[np.random.choice(pixels.shape[0], size=int(pixels.shape[0] * 0.1), replace=False)]

# Fit K-means clustering to find three main color clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=5, max_iter=50)
kmeans.fit(sampled_pixels)
print("K-means fitted.")

# Get the predicted cluster for each pixel
clusters = kmeans.predict(pixels).reshape(hsv_image.shape[:2])
print("Clusters predicted:", np.unique(clusters))

# Find the cluster centers (mean colors of clusters)
cluster_centers = kmeans.cluster_centers_
print("Cluster centers:", cluster_centers)

# Calculate the average color of each blob
average_colors = []
for cluster in np.unique(clusters):
    # Create a mask for each cluster
    mask = (clusters == cluster).astype(np.uint8) * 255
    # Calculate the average color of the pixels in this cluster
    avg_color = cv2.mean(hsv_image, mask=mask)[:3]
    average_colors.append(avg_color)

# Sort the average colors based on their brightness or V channel
average_colors = sorted(average_colors, key=lambda x: x[2])  # Sort by Value (V) channel
print("Sorted average colors based on brightness:", average_colors)

# Set thresholds based on the sorted average colors
thresholds = [average_colors[0][2], average_colors[1][2], average_colors[2][2]]
print("Automatically set thresholds:", thresholds)

# Manually assign clusters to three types based on sorted average color
light_cluster = np.argmin([color[2] for color in average_colors])   # Lightest cluster
medium_cluster = 1  # Middle cluster based on sorted average colors
dark_cluster = np.argmax([color[2] for color in average_colors])    # Darkest cluster
print(f"Sorted cluster indices: Light={light_cluster}, Medium={medium_cluster}, Dark={dark_cluster}")

# Create separate masks for each cluster type
mask_light = (clusters == light_cluster).astype(np.uint8) * 255
mask_medium = (clusters == medium_cluster).astype(np.uint8) * 255
mask_dark = (clusters == dark_cluster).astype(np.uint8) * 255
print("Masks created.")

# Apply morphological operations to clean up the masks
kernel = np.ones((2, 2), np.uint8)  # Smaller kernel for finer details
mask_light = cv2.morphologyEx(mask_light, cv2.MORPH_CLOSE, kernel)
mask_medium = cv2.morphologyEx(mask_medium, cv2.MORPH_CLOSE, kernel)
mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel)
print("Morphological operations applied.")

# Find contours for each type
contours_light, _ = cv2.findContours(mask_light, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_medium, _ = cv2.findContours(mask_medium, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_dark, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours found: Light={len(contours_light)}, Medium={len(contours_medium)}, Dark={len(contours_dark)}")

# Calculate average size of dark (tumor) cells
if len(contours_dark) > 0:
    avg_dark_size = np.mean([cv2.contourArea(c) for c in contours_dark])
else:
    avg_dark_size = 0

# Create a copy of the original image to draw contours
output_image = image.copy()
print("Created output image copy.")

# Draw smaller, more precise contours for each type
for contour in contours_light:
    if cv2.contourArea(contour) > 3:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 1:
            cv2.circle(output_image, (int(x), int(y)), int(radius), (0, 255, 0), 1)  # Green for normal cells
print("Drawn light cell contours.")

for contour in contours_medium:
    if cv2.contourArea(contour) > 3:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 1:
            cv2.circle(output_image, (int(x), int(y)), int(radius), (255, 255, 0), 1)  # Yellow for intermediate cells
print("Drawn medium cell contours.")

for contour in contours_dark:
    if cv2.contourArea(contour) > 3 and cv2.contourArea(contour) < avg_dark_size:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > 1:
            cv2.circle(output_image, (int(x), int(y)), int(radius), (0, 0, 255), 1)  # Red for tumor cells
print("Drawn dark cell contours.")

# Count the number of each type
count_light = len(contours_light)
count_medium = len(contours_medium)
count_dark = sum(1 for c in contours_dark if cv2.contourArea(c) < avg_dark_size)
print(f"Count - Normal: {count_light}, Intermediate: {count_medium}, Tumor: {count_dark}")

# Annotate the image with the count
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(output_image, f'Normal: {count_light}', (10, 30), font, 0.5, (0, 255, 0), 1)
cv2.putText(output_image, f'Intermediate: {count_medium}', (10, 50), font, 0.5, (255, 255, 0), 1)
cv2.putText(output_image, f'Tumor: {count_dark}', (10, 70), font, 0.5, (0, 0, 255), 1)
print("Image annotated with counts.")

# Save the result image with annotated contours
cv2.imwrite('output_image_with_manual_thresholds.jpg', output_image)
print("Result saved as 'output_image_with_manual_thresholds.jpg'")
