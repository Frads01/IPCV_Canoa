import cv2
import matplotlib.pyplot as plt
import numpy as np


kern = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))


def img_eq(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def hough(roi, edges):
    lines = cv2.HoughLines(edges, 0.5, np.pi / 180, 60)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return roi


def hough_p(roi, edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=50, maxLineGap=10)
    if lines is not None:
         for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return roi


def red(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([150, 30, 0]), np.array([180, 255, 255]))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kern, iterations=5)
    # cv2.imshow('mask', red)

    res = cv2.bitwise_and(roi, roi, mask=red)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    
    cv2.imshow('edges', edges)
            
    return (roi, edges)


def green(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([65, 50, 0]), np.array([130, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kern, iterations=1)
    
    cv2.imshow('mask', green)

    res = cv2.bitwise_and(roi, roi, mask=green)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)

    return (roi, edges)
    

def main():
    video = cv2.VideoCapture("cut.mp4")
    plt.ion()
    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda event: video.release())

    while True:
        ret, img = video.read()
        if not ret:
            break
        
        img = img_eq(img)

        src = img[img.shape[0]//4:img.shape[0]*3//4, img.shape[1]//6:img.shape[1]]
        
        (roi, edges) = red(src)
        # (roi, edges) = green(src)
        roi = hough(roi, edges) 
        # roi = hough_p(roi, edges)
        
        cv2.imshow('roi', roi)

        # img[img.shape[0]//4:img.shape[0]*3//4, img.shape[1]//6:img.shape[1]] = roi
        # cv2.imshow('img', img)
        q = cv2.waitKey(1) & 0xFF
        if q == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)