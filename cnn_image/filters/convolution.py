import cv2
import numpy as np

def apply_convolution_filters(img_np: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    laplacian_8bit = np.uint8(np.absolute(laplacian))

    sharpen_kernel = np.array([
        [0, -1, 0], 
        [-1, 5, -1], 
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(img_np, -1, sharpen_kernel)

    return sharpened