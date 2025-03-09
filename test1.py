import cv2
import numpy as np
from aicspylibczi import CziFile
from PIL import Image
from ultralytics import YOLO

# Step 1: Load the .czi file
print("Loading .czi file...")
file_path = '904_19_B_Ki67.czi'
czi = CziFile(file_path)
print("File loaded successfully!")

# Step 2: Read a downscaled version of the image (set channel explicitly)
print("Reading and downscaling the image...")
data = czi.read_mosaic(C=0, scale_factor=0.15)  # Use first channel (C=0) to fix error
print(f"Image shape after downscaling: {data.shape}")

# Step 3: Convert to 8-bit format and remove singleton dimensions
print("Converting image to 8-bit format...")
image = data.squeeze().astype('uint8')
print(f"Image shape after squeeze: {image.shape}")

# Step 4: Convert to RGB format (YOLO expects RGB input)
print("Converting to RGB format...")
if len(image.shape) == 2:  # If grayscale, convert to 3-channel RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Conversion to RGB successful!")

# Step 5: Save as a compressed PNG file
print("Saving compressed PNG file...")
cv2.imwrite('compressed.png', image_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print("Compressed image saved as 'compressed.png'")

# Step 6: Reload the compressed image for YOLO processing
print("Reloading compressed image for YOLO...")
image_rgb = cv2.imread('compressed.png')
if image_rgb is None:
    raise Exception("Failed to load compressed image!")
print(f"Reloaded image shape: {image_rgb.shape}")

# Step 7: Load YOLOv8 segmentation model (pre-trained)
print("Loading YOLOv8 segmentation model...")
model = YOLO('yolov8n-seg.pt')
print("Model loaded successfully!")

# Step 8: Perform segmentation using YOLOv8
print("Running segmentation on the image...")
results = model('compressed.png')  # Returns a list of results
print("Segmentation complete!")

# Step 9: Display segmented output (use the first result)
print("Displaying segmented output...")
segmented_image = results[0].plot()  # Use plot() to generate an image with annotations
cv2.imshow('Segmented Output', segmented_image)  # Show using OpenCV
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()


# Step 10: Save the segmented image
print("Saving segmented image...")
results.save('segmented_output.jpg')
print("Segmented image saved as 'segmented_output.jpg'")

# Step 11: Extract mask data for color-based manipulation
print("Extracting segmentation masks...")
masks = results[0].masks
if masks is None:
    raise Exception("No masks found!")
print(f"Number of masks found: {len(masks.data)}")

# Step 12: Create a blank image to overlay segmentation masks
print("Creating a blank image for mask overlay...")
segmented_image = np.zeros_like(image_rgb)

# Step 13: Overlay each mask with a different color
print("Applying color-based mask overlay...")
for idx, mask in enumerate(masks.data):
    # Generate a random color for each mask
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    mask = mask.cpu().numpy().astype(np.uint8) * 255
    segmented_image[mask > 0] = color
    print(f"Mask {idx + 1} applied with color {color}")

# Step 14: Save the final segmented output with mask overlay
print("Saving final segmented output...")
Image.fromarray(segmented_image).save('final_segmented_output.png')
print("Final segmented image saved as 'final_segmented_output.png'")

# Step 15: Save compressed version of the final segmented output
print("Saving compressed version of the segmented image...")
cv2.imwrite('compressed_segmented.png', segmented_image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
print("Compressed segmented image saved as 'compressed_segmented.png'")

print("✅ Process completed successfully!")
