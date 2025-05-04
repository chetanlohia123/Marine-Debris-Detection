import cv2
import numpy as np
import os

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Create a simple placeholder image
placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background

# Add Ocean Shield logo text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(placeholder, "OCEAN SHIELD", (180, 200), font, 1.5, (41, 128, 185), 2)
cv2.putText(placeholder, "Marine Debris Detection", (160, 240), font, 1, (52, 73, 94), 2)
cv2.putText(placeholder, "Upload an image or video to begin", (140, 300), font, 0.8, (100, 100, 100), 1)

# Draw a simple wave pattern at the bottom
for x in range(0, 640, 2):
    # Generate a smooth wave pattern
    y1 = int(380 + 20 * np.sin(x * 0.03))
    y2 = int(380 + 20 * np.sin((x+1) * 0.03))
    cv2.line(placeholder, (x, y1), (x+1, y2), (41, 128, 185), 2)

# Save the image
cv2.imwrite('static/images/placeholder.jpg', placeholder)

print("Placeholder image created at static/images/placeholder.jpg")