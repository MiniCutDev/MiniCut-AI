from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
model = YOLO('yolov8n.pt')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    return jsonify({'filename': f.filename})

@app.route('/track', methods=['POST'])
def track():
    filename = request.json['filename']
    in_path = os.path.join(UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(in_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out_path = os.path.join(OUTPUT_FOLDER, 'tracked_' + filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for frame in model.track(source=in_path, show=False, stream=True):
        img = frame.orig_img
        h, w = img.shape[:2]
        cx, cy = w//2, h//2

        if len(frame.boxes):
            box = frame.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ox, oy = (x1 + x2)//2, (y1 + y2)//2
            dx, dy = ox - cx, oy - cy
            start_x = max(0, min(w - w, dx + (w//2) - cx))
            start_y = max(0, min(h - h, dy + (h//2) - cy))
            cropped = img[start_y:start_y+h, start_x:start_x+w]
            frame_out = cv2.resize(cropped, (w, h))
        else:
            frame_out = img

        out.write(frame_out)

    out.release()
    return jsonify({'output': 'tracked_' + filename})

@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

