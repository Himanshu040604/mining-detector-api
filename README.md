
---

# Mining Equipment Detection API

This project is a FastAPI application that uses a YOLOv5 model to detect mining equipment (such as **Blast rig**, **Dumper truck**, **Excavator**, and **Car**) in images and videos. It includes endpoints to upload an image or a video, detect the requested class of equipment, and return the annotated results.

## Requirements

### System Requirements

* Python 3.7 or later
* Git (for version control)

### Python Dependencies

The following Python packages are required to run the project:

* `fastapi`: For building the API.
* `uvicorn`: ASGI server to run FastAPI.
* `ultralytics`: For the YOLO model and pre-trained weights.
* `opencv-python-headless`: For video processing.
* `pillow`: For image manipulation.
* `python-multipart`: For handling file uploads.
* `torch`: For running the YOLOv5 model.

### Installation

1. **Clone the repository**:

   First, clone the project repository from GitHub.

   ```bash
   git clone https://github.com/Himanshu040604/mining-detector-api.git
   cd mining-detector-api
   ```

2. **Create a virtual environment** (optional but recommended):

   It is a good practice to create a virtual environment for Python projects. You can use the following command to create one:

   ```bash
   python -m venv venv
   ```

   Then, activate the virtual environment:

   * **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   * **MacOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

3. **Install the required dependencies**:

   To install the necessary dependencies, make sure you are in the project directory (where `requirements.txt` is located) and then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:

   After the dependencies are installed, you can run the FastAPI app with `uvicorn`:

   ```bash
   uvicorn mining_api:app --reload
   ```

   The server will be running on `http://localhost:8000`.

## API Endpoints

### 1. `/detect/image/{classes}` (POST)

This endpoint allows you to upload an image and get the detection results (annotations for the specified class).

**Parameters**:

* `classes`: A comma-separated list of the class names you want to detect (e.g., `"Excavator,Dumper truck"`).
* `file`: The image file to be uploaded (JPEG/PNG).

**Response**:

* A JPG image with bounding boxes drawn for the detected objects.

**Example**:

```bash
curl -X POST "http://localhost:8000/detect/image/Excavator" \
  -F "file=@/path/to/your/image.jpg"
```

### 2. `/detect/video/{classes}` (POST)

This endpoint allows you to upload a video and get the detection results for the specified class.

**Parameters**:

* `classes`: A comma-separated list of the class names you want to detect (e.g., `"Excavator,Dumper truck"`).
* `file`: The video file to be uploaded (MP4/MOV/AVI).

**Response**:

* A video file with bounding boxes drawn for the detected objects.

**Example**:

```bash
curl -X POST "http://localhost:8000/detect/video/Excavator" \
  -F "file=@/path/to/your/video.mp4"
```

## Project Structure

```
.
├── mining_api.py      # FastAPI application code
├── requirements.txt   # Python dependencies
└── final_detection_model.pt  # YOLOv5 model file
```

## Notes

* The model weights file (`final_detection_model.pt`) is used for object detection. Ensure that it is located in the same directory as the API code.
* You may need to adjust the model and environment to run on GPUs depending on your setup.

---

