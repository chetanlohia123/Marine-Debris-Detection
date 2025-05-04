# Ocean Shield 🚱️🌊  
**Marine Debris Detection using YOLOv8 and Flask**  

---

## 📖 Overview

**Ocean Shield** is a real-time **marine debris detection system** built with **Flask** and **YOLOv8**.  
It allows users to upload images or videos, run object detection to identify marine debris, view results in a live stream, and export detection records to CSV format.  

It’s lightweight, fast, and fully customizable — designed to raise awareness and help organizations, researchers, or developers to fight marine pollution effectively.

---

## 🌟 Core Features

- 🚀 Upload and analyze **Images** and **Videos**.
- 📺 Real-time **Detection Stream** (videos).
- 🔝 **Toggle Detection** (start/pause without stopping video).
- 🧾 **Download Detections** as **CSV** (with timestamp, object class, coordinates, confidence).
- 🧹 **Clear Detections** instantly.
- 📊 Live **Latest Detections Feed** (last 50 detections shown dynamically).
- 🖼️ Beautiful **Placeholder Image** when no video is selected.

---

## 📂 Project Structure

```
OceanShield/
├── app.py                  # Main Flask application
├── best.pt                  # YOLOv8 Trained Model
├── static/
│   ├── images/
│   │   └── placeholder.jpg  # Placeholder if no video/image
│   ├── processed/           # Processed images with detection
│   └── uploads/             # Uploaded videos and images
├── templates/
│   ├── index.html           # Home page (upload)
│   ├── about.html           # About project
│   └── detection.html       # Live video detection page
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

---

## ⚙️ Quick Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ocean-shield.git
cd ocean-shield
```

### 2. Install Python Libraries

Install dependencies quickly with:
`
```bash
pip install flask opencv-python ultralytics numpy
```

Or with `requirements.txt`:

```bash
pip install -r requirements.txt
```

✅ **Ensure PyTorch is installed** separately for YOLOv8 to work properly!  
(Install instructions: [PyTorch Install](https://pytorch.org/get-started/locally/))

---

### 3. Model Placement

Ensure your YOLOv8 model file `best.pt` is placed in the project root directory.  
(You can train one using [Ultralytics YOLOv8](https://docs.ultralytics.com/modes/train/) if needed.)

---

### 4. Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:5000/
```

---

## 📥 Available API Endpoints

| Route                  | Method    | Description |
|-------------------------|-----------|-------------|
| `/`                     | GET       | Home page |
| `/about`                | GET       | About page |
| `/detection`            | GET       | Detection stream page |
| `/upload_video`         | POST      | Upload a video |
| `/upload_image`         | POST      | Upload an image |
| `/video_feed`           | GET       | Live video feed |
| `/toggle_detection`     | POST      | Toggle object detection |
| `/get_latest_detections`| GET       | Get latest 50 detections |
| `/download_csv`         | GET       | Download detections as CSV |
| `/clear_detections`     | POST      | Clear all detections |

---

## 📓 Requirements

- Python 3.8+
- Flask
- OpenCV (`opencv-python`)
- Ultralytics
- NumPy
- PyTorch

---

## ✨ Credits

- YOLOv8 by [Ultralytics](https://ultralytics.com/)
- Flask Web Framework
- OpenCV for Video and Image Processing

---

## 📜 License

This project is licensed under the MIT License — feel free to use, modify, and share it!

---

## 💬 Acknowledgements

Special thanks to open-source communities for inspiring ocean conservation initiatives. 🌊🐬

