from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
import yt_dlp


def list_formats(video_url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'listformats': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        formats = info_dict.get('formats', [])
        for fmt in formats:
            print(f"Format ID: {fmt.get('format_id')}, URL: {fmt.get('url')}")


def get_best_stream_url(video_url):
    # Define options for yt_dlp
    #list_formats(video_url)
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        formats = info_dict.get('formats', None)
        if not formats:
            raise Exception("No formats found.")

        # Choose the best format based on the criteria (e.g., video quality)
        for fmt in formats:
            if fmt.get('vcodec') != 'none' and fmt.get('acodec') != 'none':
                return fmt.get('url')

        raise Exception("No suitable formats found.")


def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return width, height


def add_text_to_image(image, text, position, font_path=None, font_size=20, text_color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Create a drawing object
    draw = ImageDraw.Draw(img_pil)

    # Load a font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Add text to the image
    draw.text(position, text, font=font, fill=text_color)

    # Convert the Pillow Image back to an OpenCV image
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return image


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov9e-seg.pt")

# Open the video file
video_path = "cut.mp4"
#video_url = 'https://www.youtube.com/watch?v=f68qtOEpxs8'
cap = cv2.VideoCapture(video_path)
#video_path = get_best_stream_url(video_url)

# Open video capture from the stream URL
cap = cv2.VideoCapture(video_path)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('output_tracking.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

scr_width, scr_height = get_screen_size()

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    roi = frame[4 * frame.shape[0] // 10:frame.shape[0], 0:frame.shape[1]]
    if success:
        # Run YOLOv9 tracking on the frame, persisting tracks between frames
        conf = 0.1
        iou = 0.5
        tracker = "bytetrack_custom.yaml"

        # Esegui l'inferenzaq
        results = model.track(roi, persist=True, conf=conf, iou=iou, show=True, classes=[0], tracker=tracker)

        if results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 160:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=5)
        else:
            annotated_frame = roi
            # Display the annotated frame

        frame[4 * frame.shape[0] // 10:frame.shape[0], 0:frame.shape[1]] = annotated_frame
        fontsize = 20
        textColor = (0, 0, 0)
        #frame = add_text_to_image(frame, time, (scr_width - 2 * fontsize, scr_height // 2), None, fontsize, textColor)
        # Write the frame to the output file
        #out.write(frame)
        cv2.imshow("YOLOv9 Tracking", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
