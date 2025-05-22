import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def angle_between(p1, p2, p3):
    """Returns angle in degrees between vectors p1p2 and p2p3."""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return np.degrees(angle)

 
def extract_most_square_region(image, debug=False):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. HSV Threshold
    lower = np.array([90, 10, 150])
    upper = np.array([150, 36, 255])
    mask = cv2.inRange(image, lower, upper)

    # 3. Morphological Close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Contour Detection
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Visualisation Setup
    output = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    all_contours_vis = output.copy()
    approx_contours_vis = output.copy()

    best_area = 0
    best_bbox = None
    best_approx = None
    approx_polys = []

    # 6. Contour Filtering
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_polys.append(approx)

        if len(approx) == 4 and cv2.contourArea(approx) >= 700 and cv2.contourArea(approx) <= 300000:
            pts = [tuple(pt[0]) for pt in approx]
            angles = [angle_between(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]

            if all(75 <= a <= 105 for a in angles):
                area = cv2.contourArea(approx)
                if area > best_area:
                    best_area = area
                    raw_x, raw_y, raw_w, raw_h = cv2.boundingRect(approx)

                    img_h, img_w = image.shape[:2]
                    x = max(raw_x - 30, 0)
                    y = max(raw_y - 30, 0)
                    w = min(raw_w + 60, img_w - x)
                    h = min(raw_h + 60, img_h - y)

                    best_bbox = (x, y, w, h)
                    best_approx = approx

    
    if (best_bbox is not None) and best_area <= 150000:
        x1,y1 = best_bbox[0], best_bbox[1]
        x2,y2 = best_bbox[0] + best_bbox[2], best_bbox[1] + best_bbox[3]

        best_bbox = (x1, y1, x2, y2)
        
    return best_bbox

def process_bbox_region2(original_img: np.ndarray, bbox: tuple, debug: bool = False):
    """
    Crops a region from the original image, thresholds for light HSV regions,
    applies morphological opening, finds the largest contour bounding box,
    then crops that box further by 20% on all sides.
 
    Parameters:
        original_img (np.ndarray): Original BGR image.
        bbox (tuple): Bounding box (x, y, w, h).
        debug (bool): Show intermediate results.
 
    Returns:
        final_crop (np.ndarray or None): The final cropped region (after 20% shrink), or None if no box found.
    """
    x, y, w, h = bbox
    cropped = original_img[y:y+h, x:x+w]
 
    # Step 1: Convert to HSV and threshold for light regions
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 10, 150])
    upper = np.array([150, 36, 255])
    mask = cv2.inRange(hsv, lower, upper)
 
    # Step 2: Morphological opening with 9x9 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 
    # Step 3: Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # no contours found
        return None, "[Process_bbox_region] No contours found]"
 
    # Step 4: Get largest contour bounding box
    largest_cnt = max(contours, key=cv2.contourArea)
    x1, y1, w1, h1 = cv2.boundingRect(largest_cnt)
 
    # Calculate 20% margin inside the bounding box
    shrink_x = int(0.2 * w1)
    shrink_y = int(0.05 * h1)
 
    # New shrunk coordinates (ensure they stay inside image bounds)
    new_x1 = max(x1 + shrink_x, 0)
    new_y1 = max(y1 + shrink_y, 0)
    new_x2 = min(x1 + w1 - shrink_x, cropped.shape[1])
    new_y2 = min(y1 + h1 - shrink_y, cropped.shape[0])
 
    # Ensure at least 1 pixel wide/high
    if new_x2 <= new_x1 or new_y2 <= new_y1:

        #print("[WARNING] Cropped region too small after 20% shrink.")

        return None, "[Process_bbox_region] Cropped region too small after shrink."
 
 
    # Perform the safe cropped region
    final_crop = cropped[new_y1:new_y2, new_x1:new_x2]
 
    # Convert to grayscale
    gray = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
 
    # Apply binary threshold
    thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)
 
    return thresh, "[Process_bbox_region] Returned binary threshold"


def extract_largest_black_object(binary_img: np.ndarray, debug: bool = False): #-> np.ndarray:
    """
    Extracts the largest black object from a binary image (black objects on white background).
 
    Parameters:
        binary_img (np.ndarray): Binary image (0 = black, 255 = white).
        debug (bool): Show debug visualizations.
 
    Returns:
        output (np.ndarray): Binary mask with only the largest black object.
    """
    # Invert: black becomes white for contour detection
    inverted = cv2.bitwise_not(binary_img)
 
    # Find contours (white blobs in the inverted image == black in original)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # print("[INFO] No contours found.")
        return None, "[INFO] No contours found."
 
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
 
    # Create a blank mask and draw the largest contour filled in black
    output = np.ones_like(binary_img) * 255  # start as white
 
    return output, ""

def resize_to_fixed_aspect(binary_image: np.ndarray, target_width: int = 1360, target_height: int = 1020) -> np.ndarray:
    """
    Resizes a binary image to fit inside a fixed 4:3 frame (default: 1360x1020) with white padding.
 
    Parameters:
        binary_image (np.ndarray): Binary (grayscale) image to resize.
        target_width (int): Target width of the output image.
        target_height (int): Target height of the output image.
 
    Returns:
        np.ndarray: Resized and centered image on white background of fixed 4:3 resolution.
    """
    h, w = binary_image.shape
 
    # Calculate scale to fit within target frame
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)
 
    new_w = int(w * scale)
    new_h = int(h * scale)
 
    # Resize image
    resized = cv2.resize(binary_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
 
    # Create white canvas and center resized image
    output = np.full((target_height, target_width), 255, dtype=np.uint8)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
 
    return output

def thicken_thin_lines(image: np.ndarray, thin_thresh: float = 1.0, kernel_size: int = 21) -> np.ndarray:
    """
    Thickens only thin black lines in a binary image (black objects on white background).
 
    Parameters:
        image (np.ndarray): Input binary image with black lines on white background.
        thin_thresh (float): Thickness threshold in pixels (distanceTransform values < this are considered thin).
        kernel_size (int): Size of the kernel used for dilation (odd number, typically 3 or 5).
 
    Returns:
        np.ndarray: Processed image with only thin black lines thickened.
    """
    # Invert: white lines on black for distance transform
    inverted = cv2.bitwise_not(image)
 
    # Ensure binary
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)
 
    # Distance transform (L2 gives smooth Euclidean distance)
    dist = cv2.distanceTransform(binary, distanceType=cv2.DIST_L2, maskSize=5)
 
    # Mask thin regions
    thin_mask = (dist > thin_thresh).astype(np.uint8) * 255
    #cv2.imshow("thinmask", thin_mask)
    # Dilate thin lines only
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_thin = cv2.dilate(thin_mask, kernel, iterations=1)
    #cv2.imshow("dilated thin", dilated_thin)
    # Combine with original binary image
    merged = cv2.bitwise_or(binary, dilated_thin)
    #cv2.imshow("merged", merged)
 
    # Invert back to original format (black lines on white)
    final = cv2.bitwise_not(merged)
 
    return final

def center_and_scale_object(binary_image, target_size=(512, 512), padding=0):
    """
    Crops, centers, and resizes a white object on black background.
   
    Args:
        binary_image (np.ndarray): Binary image with white object on black.
        target_size (tuple): Desired output size (width, height).
        padding (int): Optional padding around object in pixels.
 
    Returns:
        output (np.ndarray): Scaled and centered image.
    """
    # Find contoursfindContours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "Center and scale: No object found in the image."
        # raise ValueError("No object found in the image.")
 
    # Get bounding box of largest contour (assume it's the object)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
 
    # Add padding
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, binary_image.shape[1] - x)
    h = min(h + 2 * padding, binary_image.shape[0] - y)
 
    # Crop object
    cropped = binary_image[y:y+h, x:x+w]
 
    # Resize while maintaining aspect ratio
    aspect_ratio = w / h
    target_w, target_h = target_size
 
    if aspect_ratio > 1:
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
 
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
 
    # Center on black canvas
    output = np.zeros((target_h, target_w), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return output