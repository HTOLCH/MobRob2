#WORKING DISPLAY - READ FROM FOLDER
#and bounding BOX
#DO NOT TOUCH
 
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import itertools
 
import cv2
import numpy as np
 
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
 
 
def nothing(x):
    pass
 
def extract_largest_black_object(binary_img: np.ndarray, debug: bool = False) -> np.ndarray:
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
        if debug:
            print("[INFO] No contours found.")
        return np.zeros_like(binary_img)
 
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
 
    # Create a blank mask and draw the largest contour filled in black
    output = np.ones_like(binary_img) * 255  # start as white
    cv2.drawContours(output, [largest_contour], -1, 0, thickness=cv2.FILLED)  # fill with black
 
    if debug:
        cv2.imshow("Largest Black Object", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    return output
 
def process_bbox_region2(original_img: np.ndarray, bbox: tuple, debug: bool = True):
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
 
    if debug:
        cv2.imshow("Initial Crop", cropped)
 
    # Step 1: Convert to HSV and threshold for light regions
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower = np.array([57, 0, 190])
    upper = np.array([180, 193, 255])
    mask = cv2.inRange(hsv, lower, upper)
 
    # Step 2: Morphological opening with 9x9 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 
    if debug:
        cv2.imshow("Thresholded + Opened", opened)
 
    # Step 3: Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        if debug:
            print("[INFO] No contours found.")
        return None
 
    # Step 4: Get largest contour bounding box
    largest_cnt = max(contours, key=cv2.contourArea)
    x1, y1, w1, h1 = cv2.boundingRect(largest_cnt)
 
    if debug:
        box_vis = cropped.copy()
        cv2.rectangle(box_vis, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
        cv2.imshow("Detected Bounding Box", box_vis)
 
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
        print("[WARNING] Cropped region too small after 20% shrink.")
        return None
 
 
    # Perform the safe cropped region
    final_crop = cropped[new_y1:new_y2, new_x1:new_x2]
 
    # Convert to grayscale
    gray = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
 
    # Apply binary threshold
    thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)
 
    if debug:
        cv2.imshow("Final 20% Cropped Box", final_crop)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Thresholded", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    return thresh
 
 
def process_bbox_region(original_img: np.ndarray, bbox: tuple, debug: bool = True, min_area: int = 700):
    """
    Extracts the largest blob within a cropped region and keeps the binary mask of it.
    Also (internally) computes the region inside the blob, inclusive of any inner dark or colored areas.
    Returns the binary mask of the largest blob to maintain compatibility with existing code.
 
    Parameters:
        original_img (np.ndarray): Original BGR image.
        bbox (tuple): Bounding box in format (x, y, w, h).
        debug (bool): Whether to show intermediate images.
 
    Returns:
        largest_blob_mask (np.ndarray): Binary mask of the largest blob (white = kept region)
    """
    x, y, w, h = bbox
    cropped = original_img[y:y+h, x:x+w]
 
    if debug:
        cv2.imshow("Cropped Region", cropped)
 
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
 
    # HSV thresholding
    lower_black = np.array([0, 0, 235])
    upper_black = np.array([179, 90, 255])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
 
    lower_custom = np.array([90, 60, 160])
    upper_custom = np.array([114, 170, 255])
    mask_custom = cv2.inRange(hsv, lower_custom, upper_custom)
 
    combined_mask = cv2.bitwise_or(mask_black, mask_custom)
 
    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
 
    if num_labels <= 1:
        if debug:
            print("[INFO] No white components found.")
        return np.zeros_like(mask_black)
 
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_area = stats[largest_label, cv2.CC_STAT_AREA]
 
    if largest_area < min_area:
        if debug:
            print(f"[INFO] Largest component too small (area = {largest_area}). Ignoring.")
        return np.zeros_like(mask_black)
 
    # Create binary mask of the largest blob
    largest_blob_mask = (labels == largest_label).astype(np.uint8) * 255
 
    # Internally – fill the blob to get full interior area
    filled_mask = np.zeros_like(largest_blob_mask)
    contours, _ = cv2.findContours(largest_blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
 
    # Optional: get the full-color content inside the filled region
    masked_inside_blob = cv2.bitwise_and(cropped, cropped, mask=filled_mask)
 
    lower_custom = np.array([0, 0, 0])
    upper_custom = np.array([0, 140, 0])
    mask_custom = cv2.inRange(hsv, lower_custom, upper_custom)
 
    # ====== TODO IMPLEMENTED HERE ======
    # Show the binary values from combined_mask but only within the white region of filled_mask
    combined_inside_filled = cv2.bitwise_and(combined_mask, filled_mask)
 
    if debug:
        #cv2.imshow("Combined Mask", combined_mask)
        #cv2.imshow("Largest White Blob Only", largest_blob_mask)
        #cv2.imshow("Filled Mask", filled_mask)
        #cv2.imshow("Masked Original Inside Blob", masked_inside_blob)
        cv2.imshow("Combined Mask INSIDE Filled Region", combined_inside_filled)  # << new
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    return combined_inside_filled
 
import cv2
import numpy as np
 
def get_most_central_black_component(binary_image: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Given a binary image with black shapes on white background,
    this finds and returns the most central black component (based on centroid proximity to image center).
 
    Parameters:
        binary_image (np.ndarray): Input binary image (black = 0, white = 255).
        debug (bool): Show intermediate debug views.
 
    Returns:
        np.ndarray: A binary image with only the most central black component (black on white background).
    """
    # Invert the image so black objects become white (needed for contour finding)
    inverted = cv2.bitwise_not(binary_image)
 
    # Find external contours (black shapes)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    if not contours:
        if debug:
            print("[INFO] No contours found.")
        return np.full_like(binary_image, 255)  # all white
 
    # Get image center
    h, w = binary_image.shape
    img_center = np.array([w / 2, h / 2])
 
    min_dist = float('inf')
    best_contour = None
 
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = np.array([cx, cy])
        dist = np.linalg.norm(centroid - img_center)
        if dist < min_dist:
            min_dist = dist
            best_contour = cnt
 
    # Create output mask with only the most central black component
    output = np.full_like(binary_image, 255)  # start with white background
    if best_contour is not None:
        cv2.drawContours(output, [best_contour], -1, 0, thickness=cv2.FILLED)
 
    if debug:
        debug_vis = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_vis, [best_contour], -1, (0, 0, 255), 2)
        cv2.circle(debug_vis, tuple(img_center.astype(int)), 5, (0, 255, 0), -1)
        cv2.imshow("Debug: Most Central Black Component", debug_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    return output
 
 
def resize_and_remove_small_black_blobs(binary_image: np.ndarray, min_area: int = 100,
                                        target_size: tuple = (1360, 1020)) -> np.ndarray:
    """
    Resizes a binary image to a fixed size with white padding and removes small black blobs.
 
    Parameters:
        binary_image (np.ndarray): Input binary image (black dots on white background).
        min_area (int): Minimum area (in pixels) to retain black regions.
        target_size (tuple): Desired output image size (width, height).
 
    Returns:
        np.ndarray: Resized and cleaned binary image (white background, large black objects only).
    """
    # Ensure binary
    _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
 
    # Resize with aspect ratio preserved and white padding
    h, w = binary.shape
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
 
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
 
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)  # white background
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
 
    # Invert so black blobs become white for connected component analysis
    inverted = cv2.bitwise_not(canvas)
 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
 
    # Rebuild cleaned mask: only keep large enough white blobs (i.e., original black dots)
    clean_mask = np.zeros_like(inverted)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
 
    # Invert back to get black on white
    cleaned = cv2.bitwise_not(clean_mask)
    return cleaned
 
 
def isolate_eroded_largest_contour(binary_image: np.ndarray, erosion_size: int = 9, debug: bool = False) -> np.ndarray:
    """
    Morphologically closes the binary image, extracts the white region as a mask,
    erodes the mask, and pastes only the content inside the eroded region from the original
    binary image onto a white background.
 
    Parameters:
        binary_image (np.ndarray): Binary image with white object and black content on black background.
        erosion_size (int): Size of square kernel for erosion (default 15).
        debug (bool): Show intermediate steps if True.
 
    Returns:
        np.ndarray: Output image with only eroded white region content on a white background.
    """
 
    # Step 1: Clone binary image and apply morphological closing
    closed = binary_image.copy()
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_close)
 
    if debug:
        cv2.imshow("Closed Image (White Mask Source)", closed)
 
    # Step 2: Create mask where closed image is white
    white_mask = np.zeros_like(binary_image)
    white_mask[closed == 255] = 255
 
    if debug:
        cv2.imshow("Initial White Mask", white_mask)
 
    # Step 3: Erode the white mask
    kernel_erode = np.ones((erosion_size, erosion_size), dtype=np.uint8)
    eroded_mask = cv2.erode(white_mask, kernel_erode, iterations=1)
 
    if debug:
        cv2.imshow("Eroded Mask", eroded_mask)
 
    # Step 4: Copy only content inside the eroded mask from the original image
    output = np.full_like(binary_image, 255)  # white background
    output[eroded_mask == 255] = binary_image[eroded_mask == 255]
 
    if debug:
        cv2.imshow("Final Output (Masked Content on White)", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    return output
 
 
def connect_all_components_until_single(binary_image, line_thickness=2, debug=False, max_iter=50):
    """
    Repeatedly connects the two closest components in a binary image until only one black component remains.
 
    Args:
        binary_image (np.ndarray): A binary image (black digits on white background).
        line_thickness (int): Thickness of the connecting lines.
        debug (bool): If True, displays intermediate visualizations.
        max_iter (int): Safety cap on number of iterations.
 
    Returns:
        np.ndarray: Modified binary image with a single connected component.
    """
    # Copy the image
    connected = binary_image.copy()
 
    for step in range(max_iter):
        # Invert: black becomes white for contour detection
        inverted = cv2.bitwise_not(connected)
 
        # Find contours
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        if len(contours) <= 1:
            if debug:
                print(f"[INFO] Single component achieved at iteration {step}.")
            break
 
        # Find closest pair of contours
        min_dist = float('inf')
        closest_pair = None
 
        for c1, c2 in itertools.combinations(contours, 2):
            for p1 in c1:
                for p2 in c2:
                    dist = np.linalg.norm(p1[0] - p2[0])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (tuple(p1[0]), tuple(p2[0]))
 
        # Draw line to connect the closest components
        if closest_pair:
            cv2.line(connected, closest_pair[0], closest_pair[1], color=0, thickness=line_thickness)
 
        if debug:
            debug_vis = cv2.cvtColor(connected, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_vis, closest_pair[0], 3, (0, 0, 255), -1)
            cv2.circle(debug_vis, closest_pair[1], 3, (0, 0, 255), -1)
            #cv2.imshow(f"Step {step} - Connecting", debug_vis)
 
    else:
        if debug:
            print("[WARNING] Reached max iterations without merging all components.")
 
    #if debug:
        #cv2.destroyAllWindows()
 
    return connected
 
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
 
 
def angle_between(p1, p2, p3):
    """Return angle in degrees between vectors p1p2 and p2p3."""
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
    return np.degrees(angle)


def angle_between(p1, p2, p3):
        """Returns angle in degrees between vectors p1p2 and p2p3."""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        angle = np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))
        return np.degrees(angle)

 
def extract_most_square_region(image, debug=False):
 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 2. HSV Threshold
    lower = np.array([106, 16, 167])
    upper = np.array([114, 125, 255])
    mask = cv2.inRange(image, lower, upper)
 
    # 3. Morphological Close
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 
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
 
        if len(approx) == 4 and cv2.contourArea(approx) >= 700:
            pts = [tuple(pt[0]) for pt in approx]
            angles = [angle_between(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]
 
            if all(70 <= a <= 110 for a in angles):
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

    return best_bbox
 
 
 
# ===== Model Definition =====
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
 
 
# ===== Function to Predict Digit =====
# ===== Function to Predict Digit =====
def predict_emnist_digit(centered_image, model_path="emnist_digits_cnn.pth", show=True):
    
    resized = cv2.resize(centered_image, (28, 28), interpolation=cv2.INTER_AREA)
    digit_resized = cv2.resize(resized, (24, 24), interpolation=cv2.INTER_AREA)
 
    canvas = np.zeros((28, 28), dtype=np.uint8)
    offset = (28 - 24) // 2
    canvas[offset:offset+24, offset:offset+24] = digit_resized
 
    canvas[0, :] = canvas[-1, :] = 0
    canvas[:, 0] = canvas[:, -1] = 0
 
    # 🔁 EMNIST FIX: Rotate image to match training orientation
 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = transform(canvas).unsqueeze(0)
 
    model = SimpleCNN()
    model.load_state_dict(torch.load("emnist_digits_cnn.pth"))
    model.eval()
 
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()
 
    if show:
        plt.imshow(img_tensor.squeeze().numpy(), cmap='gray')
        plt.title(f"Predicted Digit: {pred} (Conf: {confidence:.2f})")
        plt.axis('off')
        plt.show()
 
    return pred, confidence
 
 
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
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No object found in the image.")
 
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
 
import glob
 
skipped_images = []
 
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return None, None, None
 
    try:
        scale = 0.5
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        orig_display = cv2.resize(img.copy(), (256, 256))  # Save for final grid
 
        height = img.shape[0]
        cut = int(0.3 * height)
        img = img[cut:, :]
 
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 190])
        upper_white = np.array([180, 90, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
 
        result = extract_most_square_region(img, debug=False)
        if not result:
            skipped_images.append(os.path.basename(img_path))
            return None, None, None
 
 
        x, y, w, h = result
        #--------------------------------------------------------------------------------------------------------------------------------------------
        binary_output = process_bbox_region2(img, (x, y, w, h), debug=False)
        #process_bbox_region_with_sliders(img, (x, y, w, h), debug=True)
 
        binary_output = extract_largest_black_object(binary_output, debug=False)
 
        #binary_output = resize_and_remove_small_black_blobs(binary_output, min_area=5)
 
        #binary_output = isolate_eroded_largest_contour(binary_output, debug=True)
       
        #binary_output = connect_all_components_until_single(binary_output, line_thickness=2)
        #binary_output = get_most_central_black_component(binary_output, debug=False)
        binary_output = resize_to_fixed_aspect(binary_output)
        #cv2.imshow("resize", binary_output)
        binary_output = thicken_thin_lines(binary_output)
        #cv2.imshow("resize2", binary_output)
 
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
 
        # Isolate largest black component
        inverted = cv2.bitwise_not(binary_output)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(binary_output)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
            final_output = np.full_like(binary_output, 255)
            final_output[mask == 255] = binary_output[mask == 255]
        else:
            final_output = binary_output.copy()
 
        # Final center & scale
        inv = cv2.bitwise_not(final_output)
        centered = center_and_scale_object(inv, target_size=(512, 512), padding=60)
        # Increase brightness of centered image
        #bright_mask = centered > 180  # tune threshold as needed
        #centered[bright_mask] = np.clip(centered[bright_mask] + 50, 0, 255)
 
 
        if np.all(centered == 0) or np.all(centered == 255):
            skipped_images.append(os.path.basename(img_path))
            return None, None, None
 
        else:
            pred, conf = predict_emnist_digit(centered, model_path="emnist_digits_cnn.pth")
            if conf >= 0.85 and result:
                x_box, y_box, w_box, h_box = result
 
                label_text = f"{pred} ({conf:.2f})"
                cv2.rectangle(img, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 3)
                cv2.putText(img, label_text, (x_box, y_box - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
 
                print(f"[✔] High Confidence Detection: {label_text} at (x={x_box}, y={y_box}, w={w_box}, h={h_box})")
                cv2.imshow("Detected + Labeled (Aligned with Extract)", img)
 
 
 
 
        # Resize and annotate for display
        final_canvas = cv2.resize(centered, (256, 256))
        annotated = cv2.putText(final_canvas.copy(), label, (10, 245),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
 
        return orig_display, annotated, label
 
    except Exception as e:
        print(f"[ERROR] Processing {img_path} failed: {e}")
        skipped_images.append(os.path.basename(img_path))
        return None, None, None
 
 
# ==== READ FROM FOLDER ====
folder = "Newdata"
image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
 
original_images = []
processed_images = []
labels = []
 
for path in image_paths:
    print(path)
    orig, pred_img, label = process_image(path)
    if orig is not None and label != "Blank":  # ✅ Only include if prediction was made
        original_images.append(cv2.resize(orig, (256, 256)))
        label_img = cv2.putText(pred_img.copy(), label, (10, 500),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 2)
        processed_images.append(cv2.resize(label_img, (256, 256)))
        labels.append(label)
 
rows = []
num_images = len(original_images)
cols = math.ceil(num_images / 4)  # automatically determine column count
 
for row_idx in range(4):
    orig_row = []
    for col_idx in range(cols):
        idx = row_idx * cols + col_idx
        if idx < num_images:
            orig_img = original_images[idx]
            proc_img = processed_images[idx]
 
            if len(proc_img.shape) == 2:
                proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
 
            # Stack original + prediction side by side
            pair = np.hstack((orig_img, proc_img))
        else:
            # Pad with blank image if not enough to fill grid
            pair = np.hstack((
                np.full((256, 256, 3), 255, dtype=np.uint8),
                np.full((256, 256, 3), 255, dtype=np.uint8)
            ))
 
        orig_row.append(pair)
 
    # Stack all pairs horizontally to form a full row
    if orig_row:
        rows.append(np.hstack(orig_row))
    else:
        print("[WARNING] Skipping row: no valid images to stack.")
 
 
# Stack all rows vertically to form the final grid
final_grid = np.vstack(rows)
 
final_scaled = cv2.resize(final_grid, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
cv2.imshow("All Inputs & Predictions (4-row grid)", final_scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
print("\n=== Skipped Images ===")
if not skipped_images:
    print("None! All images produced predictions.")
else:
    for name in skipped_images:
        print(f"- {name}")