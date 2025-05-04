from flask import Flask, render_template, Response, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import time
import os
import csv
import json
import threading
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load the trained model
model = YOLO("best.pt")  # Ensure 'best.pt' is in the same directory

# Global variables
video_capture = None
output_frame = None
detect_objects = False
processing_frame = False
all_detections = []  # Store all detections for CSV export
detection_lock = threading.Lock()

def detect_and_display(frame):
    """
    Process frame with YOLO model and return annotated frame
    """
    global processing_frame, all_detections
    
    if processing_frame:
        return frame, []
    
    processing_frame = True
    
    # Run inference
    results = model(frame)
    
    # Get the annotated frame with detections
    annotated_frame = results[0].plot()  # Automatically adds boxes and labels
    
    # Extract detection results
    frame_detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            
            detection = {
                'class': cls_name,
                'confidence': conf,
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            }
            
            frame_detections.append(detection)
    
    # Add current timestamp to the frame
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(annotated_frame, current_time, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Update all detections for CSV export
    with detection_lock:
        all_detections.extend(frame_detections)
    
    processing_frame = False
    return annotated_frame, frame_detections

def generate_frames():
    """
    Generator function to yield processed frames
    """
    global output_frame, video_capture, detect_objects
    
    # Target FPS
    target_fps = 6
    frame_time = 1 / target_fps
    
    while True:
        if video_capture is None:
            # Return a blank frame if no video source is initialized
            blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(blank_frame, "No video selected", (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        
        start_time = time.time()  # Track start time of frame processing
        
        success, frame = video_capture.read()
        if not success:
            # If we've reached the end of the video file, loop back to the beginning
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = video_capture.read()
            if not success:
                blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
                cv2.putText(blank_frame, "Error reading video", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue
        
        if detect_objects:
            # Apply object detection
            try:
                processed_frame, _ = detect_and_display(frame)
                output_frame = processed_frame
            except Exception as e:
                print(f"Error in object detection: {e}")
                output_frame = frame
        else:
            output_frame = frame
            # Add timestamp to the frame
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(output_frame, current_time, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        
        # Calculate elapsed time and wait if necessary to maintain target fps
        elapsed_time = time.time() - start_time
        sleep_time = max(0, frame_time - elapsed_time)
        time.sleep(sleep_time)
        
        # Yield the frame in the format expected by MultipartResponse
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/detection')
def detection():
    """Render the detection page"""
    return render_template('detection.html')

@app.route('/video_feed')
def video_feed():
    """Route for streaming the video feed"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload and process a video file"""
    global video_capture, all_detections
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    # Save the file temporarily
    video_path = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    file.save(video_path)
    
    # Close any existing video capture
    if video_capture is not None:
        video_capture.release()
    
    # Reset detections list
    with detection_lock:
        all_detections = []
    
    # Open the video
    video_capture = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video_capture.isOpened():
        return jsonify({'status': 'error', 'message': f'Could not open video: {file.filename}'})
    
    return jsonify({
        'status': 'success',
        'message': f'Video uploaded successfully: {file.filename}',
        'video_path': video_path
    })

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle object detection on/off"""
    global detect_objects
    
    detect_objects = not detect_objects
    status = 'enabled' if detect_objects else 'disabled'
    
    return jsonify({'status': 'success', 'detection': status})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Process an uploaded image"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    # Save the file temporarily
    temp_path = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    file.save(temp_path)
    
    # Process the image
    frame = cv2.imread(temp_path)
    if frame is None:
        return jsonify({'status': 'error', 'message': 'Could not read image'})
    
    processed_frame, detections = detect_and_display(frame)
    
    # Save the processed image
    output_path = os.path.join('static', 'processed', file.filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, processed_frame)
    
    # Return the path to the processed image and detections
    return jsonify({
        'status': 'success', 
        'image_path': f'/static/processed/{file.filename}',
        'detections': detections
    })

@app.route('/get_latest_detections', methods=['GET'])
def get_latest_detections():
    """Return the latest detection results"""
    global all_detections
    
    with detection_lock:
        detections = all_detections[-50:] if len(all_detections) > 50 else all_detections.copy()
    
    return jsonify({
        'status': 'success',
        'detections': detections
    })

@app.route('/download_csv', methods=['GET'])
def download_csv():
    """Generate and download CSV file with all detections"""
    global all_detections
    
    # Create a CSV file
    csv_path = os.path.join('static', 'detections.csv')
    
    with detection_lock:
        detections_copy = all_detections.copy()
    
    if not detections_copy:
        return jsonify({'status': 'error', 'message': 'No detections to download'})
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for detection in detections_copy:
            writer.writerow({
                'timestamp': detection['timestamp'],
                'class': detection['class'],
                'confidence': detection['confidence'],
                'x1': detection['box'][0],
                'y1': detection['box'][1],
                'x2': detection['box'][2],
                'y2': detection['box'][3]
            })
    
    return send_file(csv_path, as_attachment=True, download_name='detections.csv')

@app.route('/clear_detections', methods=['POST'])
def clear_detections():
    """Clear all stored detections"""
    global all_detections
    
    with detection_lock:
        all_detections = []
    
    return jsonify({'status': 'success', 'message': 'All detections cleared'})

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/processed', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    # Create placeholder image if it doesn't exist
    placeholder_path = os.path.join('static', 'images', 'placeholder.jpg')
    if not os.path.exists(placeholder_path):
        # Create a simple placeholder image
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add Ocean Shield logo text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(placeholder, "OCEAN SHIELD", (180, 200), font, 1.5, (8, 145, 178), 2)
        cv2.putText(placeholder, "Marine Debris Detection", (160, 240), font, 1, (22, 78, 99), 2)
        cv2.putText(placeholder, "Upload an image or video to begin", (140, 300), font, 0.8, (100, 100, 100), 1)
        
        # Draw a simple wave pattern at the bottom
        for x in range(0, 640, 2):
            # Generate a smooth wave pattern
            y1 = int(380 + 20 * np.sin(x * 0.03))
            y2 = int(380 + 20 * np.sin((x+1) * 0.03))
            cv2.line(placeholder, (x, y1), (x+1, y2), (8, 145, 178), 2)
            
        cv2.imwrite(placeholder_path, placeholder)
        print(f"Created placeholder image at {placeholder_path}")
    
app.run(debug=True, host='0.0.0.0', port=5001)
