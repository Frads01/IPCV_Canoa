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


# read the image
image = cv2.imread("Canoa3.png")
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Split the LAB image to different channels
l, a, b = cv2.split(lab_image)

# Apply histogram equalization to the L channel
l_equalized = cv2.equalizeHist(l)

# Merge the equalized L channel back with A and B channels
lab_equalized = cv2.merge((l_equalized, a, b))

# Convert the LAB image back to BGR color space
equalized_image = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
result_img, _ = predict_and_detect(model, equalized_image, classes=[], conf=0.4)
cv2.imshow("Result", result_img)
#model.export(format="onnx")
cv2.waitKey(0)
