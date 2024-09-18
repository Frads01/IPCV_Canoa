import cv2
from ultralytics import YOLO

model = YOLO("yolov9e.pt")


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes is None:
        classes = []
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results


url = 'https://www.youtube.com/watch?v=f68qtOEpxs8' #direct camera
url = 'https://cdn.top-ix.org/ivreacanoa/streaming_out_1080p_2024-07-17-08.07.49.494-UTC_2.mp4' #recording
# Open the video stream
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Error: Could not open video stream")
ret, frame = cap.read()
cv2.imshow('Video', frame)
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (frame.shape[1], frame.shape[0]), True)
# Define the codec and create VideoWriter object

while True:

    if not ret:
        break
    #cv2.imshow('frame', frame)
    height, width, channels = frame.shape

    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split the LAB image to different channels
    l, a, b = cv2.split(lab_image)

    # Apply histogram equalization to the L channel
    l_equalized = cv2.equalizeHist(l)

    # Merge the equalized L channel back with A and B channels
    lab_equalized = cv2.merge((l_equalized, a, b))

    # Convert the LAB image back to BGR color space
    equalized_image = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    result_img, _ = predict_and_detect(model, equalized_image, classes=[], conf=0.4)
    out.write(result_img)
    # Display the resulting frame
    #cv2.imshow('Identification', result_img)
    ret, frame = cap.read()
    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
