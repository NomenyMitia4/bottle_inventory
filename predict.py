import cv2
from PIL import Image
import numpy as np
import webcolors
import colorsys

from ultralytics.models.yolo.model import YOLO

model = YOLO("yolo11n.pt")

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model("./images/bottle.jpg") 
# results = model.predict(source="0")
# results = model.predict(source="./", show=True)  # Display preds. Accepts all YOLO predict arguments

# Read an image using OpenCV
source = cv2.imread("./images/tavoahangy.jpg")

# Run inference on the source
results = model.predict(source, classes=[39], retina_masks=True)  # list of Results objects

number_of_bottle = 0

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    print(masks)
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
        # Check if masks are available
    if masks is not None:
        # Get the mask for the first detected object (assuming it's a bottle)
        mask = masks[0].data[0].cpu().numpy()  # Convert mask to numpy array

        # Resize the mask to match the original image dimensions
        mask = cv2.resize(mask, (source.shape[1], source.shape[0]))

        # Convert the mask to a binary mask (0s and 1s)
        mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold and scale to 0-255

        # Create a colored mask for visualization
        color_mask = np.zeros_like(source)
        color_mask[mask == 255] = [0, 255, 0]  # Green color for the mask

        # Overlay the mask on the original image
        overlayed_image = cv2.addWeighted(source, 0.7, color_mask, 0.3, 0)

        # Save the overlayed image
        cv2.imwrite("./result/result_with_mask.jpg", overlayed_image)
        
    result.save(filename=f"./result/result_prediction.jpg")  # save to disk
    
number_of_bottle = len(boxes)
print("Isan'ny tavoahangy: "+str(number_of_bottle))

def rgb_to_color_name(rgb):
    min_colours = {}
    color_name = "None"
    try:
        # Try to get the exact color name
        color_name = webcolors.rgb_to_name(rgb)
    except ValueError as e:
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - rgb[0]) ** 2
            gd = (g_c - rgb[1]) ** 2
            bd = (b_c - rgb[2]) ** 2
            min_colours[(rd + gd + bd)] = name
    color_name = min_colours[min(min_colours.keys())]
    return color_name

for i, box in enumerate(boxes):
    # Get the bounding box coordinates (x1, y1, x2, y2)
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert tensor to integers

    # Crop the bottle region from the image
    bottle_frame = source[y1:y2, x1:x2]
    x, y, width, height = 100, 100, 300, 200  # Example values
    
    # Define the capsule region (top 20% of the bottle's bounding box)
    capsule_height = int((y2 - y1) * 0.1)  # 20% of the bottle height

    capsule_x1, capsule_y1 = 200, 0  # Top-left corner of the capsule region
    capsule_x2, capsule_y2 = (x2-200) - x1, capsule_height  # Bottom-right corner of the capsule region
    
    # Crop the capsule region
    capsule_frame = bottle_frame[capsule_y1:capsule_y2, capsule_x1:capsule_x2]
    
    # Save the cropped bottle image and capsule image
    print("saving image...")
    cv2.imwrite(f"./bottle_frames/bottle_{i + 1}.jpg", bottle_frame)
    cv2.imwrite(f"./capsule_frames/capsule_{i + 1}.jpg", capsule_frame)
    
    # Check the capsule color
    # Convert the capsule region to HSV color space
    hsv_capsule = cv2.cvtColor(capsule_frame, cv2.COLOR_BGR2HSV)

    # Calculate the average color of the capsule region
    hsv_color = np.mean(hsv_capsule, axis=(0, 1))
    
    # Convert HSV to RGB
    rgb_color = colorsys.hsv_to_rgb(hsv_color[0] / 179, hsv_color[1] / 255, hsv_color[2] / 255)
    rgb_color = tuple(int(c * 255) for c in rgb_color)  # Scale to 0-255
    
    # average_color = np.array(average_color)
    capsule_color = rgb_to_color_name(rgb_color)
    
    print(f"Capsule {i + 1} average color (HSV): {capsule_color}")
    print("Color of te capsule: "+str(capsule_color))
