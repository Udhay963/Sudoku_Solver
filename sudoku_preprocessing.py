import cv2
import numpy as np

def order_points(pts):
    """ Orders the points in a consistent order: top-left, top-right, bottom-right, bottom-left. """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to get ordered points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        ordered_points = order_points(approx.reshape(4, 2))  # Correct order
    else:
        raise ValueError("Could not find a proper Sudoku grid.")

    size = 450
    destination_points = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_points, destination_points)
    warped = cv2.warpPerspective(gray, matrix, (size, size))

    return warped

# Example usage
sudoku_image = "sudoku1.jpg"  # Replace with your image
processed_image = preprocess_image(sudoku_image)
cv2.imwrite("processed_image.jpg", processed_image)
