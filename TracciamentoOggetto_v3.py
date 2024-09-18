import threading
from collections import defaultdict
import cv2
import tkinter as tk
from enum import Enum 

from cv2 import Mat
from ultralytics import YOLO
import numpy as np


class Entrata(Enum): # Verso di ENTRATA nella porta
    ALTO = 0,
    BASSO = 1,


class Porta:
    color: (int, int, int)
    numero: int
    tipo: Entrata
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float

    def __init__(self, x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0, color=(255, 255, 255), numero=0, tipo=Entrata.ALTO):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4
        self.color = color
        self.numero = numero
        self.tipo = tipo

    def new(self, x1, y1, x2, y2, x3, y3, x4, y4, color: tuple, numero=0, tipo: Entrata = Entrata.ALTO):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4
        self.color = color
        self.numero = numero
        self.tipo = tipo

    def draw(self, img) -> Mat | np.ndarray:
        pts = np.array([[self.x1, self.y1], [self.x2, self.y2], [self.x4, self.y4], [self.x3, self.y3]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Disegna il rettangolo sull'immagine usando cv2.polylines()
        cv2.polylines(img, [pts], isClosed=True, color=self.color, thickness=3, lineType=cv2.LINE_8)
        # cv2.rectangle(img, (self.x1, self.y1), (self.x3, self.y3), self.color, 3)
        return img

    def width(self):
        return cv2.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    def height(self):
        return self.y4 - self.y2


OFFSET = 20
FRAME_PRECEDENTI = 3
RED = (0, 0, 255)
GREEN = (0, 255, 0)
PORTE_Inizio: list[Porta] = [
    Porta(x1=153, y1=419, x2=185, y2=415, x3=156, y3=468, x4=186, y4=461, color=GREEN, numero=1, tipo=Entrata.ALTO),
    Porta(x1=279, y1=448, x2=308, y2=438, x3=281, y3=488, x4=309, y4=474, color=GREEN, numero=2, tipo=Entrata.ALTO),
    Porta(x1=270, y1=552, x2=310, y2=527, x3=272, y3=608, x4=310, y4=582, color=GREEN, numero=3, tipo=Entrata.ALTO),
    Porta(x1=386, y1=438, x2=412, y2=431, x3=388, y3=478, x4=412, y4=466, color=RED, numero=4, tipo=Entrata.ALTO),
    Porta(x1=451, y1=669, x2=484, y2=629, x3=453, y3=727, x4=483, y4=684, color=GREEN, numero=5, tipo=Entrata.ALTO),
    
    Porta(x1=680, y1=525, x2=678, y2=498, x3=677, y3=564, x4=676, y4=540, color=RED, numero=6, tipo=Entrata.BASSO),   # @TODO
    Porta(x1=889, y1=614, x2=857, y2=585, x3=883, y3=659, x4=855, y4=627, color=GREEN, numero=6, tipo=Entrata.BASSO), # 6 o 7?
    Porta(x1=1089, y1=575, x2=1064, y2=550, x3=1085, y3=613, x4=1059, y4=589, color=GREEN, numero=7, tipo=Entrata.BASSO),
    Porta(x1=1114, y1=514, x2=1093, y2=499, x3=1109, y3=553, x4=1090, y4=537, color=RED, numero=8, tipo=Entrata.BASSO),
    Porta(x1=1253, y1=587, x2=1218, y2=558, x3=1247, y3=617, x4=1212, y4=602, color=GREEN, numero=9, tipo=Entrata.BASSO),
    Porta(x1=1208, y1=522, x2=1182, y2=505, x3=1203, y3=557, x4=1178, y4=539, color=RED, numero=10, tipo=Entrata.BASSO),
    Porta(x1=1221, y1=469, x2=1205, y2=466, x3=1217, y3=506, x4=1203, y4=494, color=RED, numero=11, tipo=Entrata.BASSO),
    Porta(x1=1342, y1=500, x2=1320, y2=488, x3=1337, y3=534, x4=1315, y4=523, color=GREEN, numero=12, tipo=Entrata.BASSO),
    Porta(x1=1400, y1=531, x2=1371, y2=529, x3=1396, y3=566, x4=1367, y4=558, color=GREEN, numero=13, tipo=Entrata.BASSO),
    Porta(x1=1448, y1=478, x2=1426, y2=470, x3=1444, y3=509, x4=1424, y4=501, color=GREEN, numero=14, tipo=Entrata.BASSO),
    Porta(x1=1459, y1=451, x2=1437, y2=445, x3=1456, y3=447, x4=1434, y4=471, color=GREEN, numero=15, tipo=Entrata.BASSO),
    Porta(x1=1498, y1=438, x2=1478, y2=433, x3=1496, y3=462, x4=1476, y4=454, color=GREEN, numero=16, tipo=Entrata.BASSO)
]
PORTE_PonteDestra: list[Porta] = [
    Porta(x1=1342, y1=587, x2=1491, y2=503, x3=1331, y3=770, x4=1468, y4=740, color=GREEN, numero=30, tipo=Entrata.ALTO),
    Porta(x1=1526, y1=489, x2=1600, y2=470, x3=1517, y3=653, x4=1594, y4=630, color=GREEN, numero=29, tipo=Entrata.ALTO),
    Porta(x1=1219, y1=500, x2=1321, y2=465, x3=1209, y3=662, x4=1316, y4=636, color=GREEN, numero=28, tipo=Entrata.ALTO),
    Porta(x1=658, y1=454, x2=805, y2=454, x3=664, y3=619, x4=805, y4=607, color=RED, numero=27, tipo=Entrata.ALTO),
    Porta(x1=610, y1=460, x2=720, y2=452, x3=614, y3=587, x4=723, y4=577, color=GREEN, numero=26, tipo=Entrata.ALTO),
    Porta(x1=1015, y1=443, x2=1090, y2=412, x3=1015, y3=573, x4=1090, y4=519, color=GREEN, numero=25, tipo=Entrata.ALTO),
    Porta(x1=474, y1=429, x2=576, y2=435, x3=474, y3=543, x4=576, y4=546, color=RED, numero=24, tipo=Entrata.ALTO)
]
PORTE_PonteSinistra: list[Porta] = [
    Porta(x1=1170, y1=228, x2=519, y2=231, x3=1139, y3=842, x4=575, y4=915, color=GREEN, numero=31, tipo=Entrata.BASSO),
    Porta(x1=935, y1=144, x2=1239, y2=233, x3=1226, y3=429, x4=941, y4=462, color=RED, numero=33, tipo=Entrata.BASSO),
    Porta(x1=1596, y1=138, x2=1406, y2=120, x3=1580, y3=338, x4=1398, y4=339, color=GREEN, numero=34, tipo=Entrata.BASSO),
    Porta(x1=508, y1=185, x2=343, y2=205, x3=511, y3=324, x4=350, y4=340, color=GREEN, numero=35, tipo=Entrata.BASSO),
    Porta(x1=1087, y1=153, x2=981, y2=165, x3=1086, y3=286, x4=982, y4=298, color=GREEN, numero=36, tipo=Entrata.BASSO),
    Porta(x1=971, y1=155, x2=855, y2=160, x3=971, y3=271, x4=858, y4=277, color=GREEN, numero=37, tipo=Entrata.BASSO),
    Porta(x1=667, y1=156, x2=573, y2=160, x3=667, y3=279, x4=577, y4=279, color=RED, numero=38, tipo=Entrata.BASSO),
    Porta(x1=261, y1=163, x2=154, y2=168, x3=265, y3=277, x4=158, y4=279, color=RED, numero=39, tipo=Entrata.BASSO),
    Porta(x1=743, y1=124, x2=658, y2=121, x3=745, y3=209, x4=657, y4=210, color=GREEN, numero=40, tipo=Entrata.BASSO),
]
PORTE_Balcone: list[Porta] = [
    Porta(x1=403, y1=533, x2=303, y2=532, x3=428, y3=744, x4=322, y4=722, color=GREEN, numero=41, tipo=Entrata.BASSO),
    Porta(x1=789, y1=458, x2=642, y2=457, x3=816, y3=712, x4=662, y4=704, color=GREEN, numero=42, tipo=Entrata.BASSO),
    Porta(x1=1035, y1=449, x2=910, y2=447, x3=1045, y3=623, x4=918, y4=607, color=GREEN, numero=43, tipo=Entrata.BASSO),
    Porta(x1=1354, y1=436, x2=1238, y2=420, x3=1358, y3=578, x4=1242, y4=567, color=GREEN, numero=44, tipo=Entrata.BASSO),
    Porta(x1=1784, y1=445, x2=1663, y2=437, x3=1787, y3=558, x4=1666, y4=566, color=RED, numero=45, tipo=Entrata.BASSO),
    Porta(x1=1687, y1=419, x2=1594, y2=424, x3=1686, y3=535, x4=1594, y4=531, color=RED, numero=46, tipo=Entrata.BASSO),
    Porta(x1=1497, y1=411, x2=1411, y2=415, x3=1502, y3=512, x4=1414, y4=508, color=GREEN, numero=47, tipo=Entrata.BASSO),
]
PORTE_LungoCanale: list[Porta] = [
    Porta(x1=382, y1=430, x2=367, y2=339, x3=386, y3=507, x4=365, y4=471, color=GREEN, numero=50, tipo=Entrata.BASSO),
    Porta(x1=866, y1=465, x2=796, y2=423, x3=853, y3=640, x4=784, y4=594, color=RED, numero=51, tipo=Entrata.BASSO),
    Porta(x1=1203, y1=411, x2=1122, y2=374, x3=1186, y3=526, x4=1111, y4=490, color=GREEN, numero=52, tipo=Entrata.BASSO),
    Porta(x1=1291, y1=319, x2=1234, y2=309, x3=1278, y3=404, x4=1222, y4=390, color=RED, numero=53, tipo=Entrata.BASSO),
    Porta(x1=1529, y1=400, x2=1444, y2=375, x3=1514, y3=499, x4=1430, y4=473, color=GREEN, numero=54, tipo=Entrata.BASSO),
    Porta(x1=1604, y1=379, x2=1554, y2=362, x3=1586, y3=463, x4=1538, y4=443, color=GREEN, numero=55, tipo=Entrata.BASSO),
]
PORTE_Arrivo: list[Porta] = [
    Porta(x1=361, y1=260, x2=436, y2=247, x3=364, y3=331, x4=437, y4=327, color=RED, numero=52, tipo=Entrata.ALTO),
    Porta(x1=429, y1=293, x2=504, y2=273, x3=429, y3=403, x4=504, y4=382, color=GREEN, numero=54, tipo=Entrata.ALTO),
    Porta(x1=869, y1=268, x2=922, y2=263, x3=862, y3=372, x4=917, y4=358, color=GREEN, numero=55, tipo=Entrata.ALTO),
    Porta(x1=873, y1=377, x2=981, y2=364, x3=866, y3=522, x4=965, y4=503, color=RED, numero=56, tipo=Entrata.ALTO),
    Porta(x1=1275, y1=514, x2=1355, y2=487, x3=1248, y3=658, x4=1334, y4=616, color=GREEN, numero=57, tipo=Entrata.ALTO),
    Porta(x1=1201, y1=584, x2=1329, y2=547, x3=1175, y3=782, x4=1290, y4=738, color=GREEN, numero=58, tipo=Entrata.ALTO),
    Porta(x1=1721, y1=482, x2=1750, y2=456, x3=1694, y3=598, x4=1727, y4=565, color=GREEN, numero=59, tipo=Entrata.ALTO),
    Porta(x1=1718, y1=674, x2=1747, y2=635, x3=1652, y3=853, x4=1697, y4=800, color=GREEN, numero=60, tipo=Entrata.ALTO),
]


def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return width, height


def check(track: list[any]):
    track_rev = track.copy()
    track_rev.reverse()
    for porta in PORTE_PonteDestra:
        (xm, ym) = (porta.x3+porta.x4)/2, (porta.y3+porta.y4)/2
        
        # min e max
        [min_x, max_x] = [min(porta.x3, porta.x4), max(porta.x3, porta.x4)]
        [min_y, max_y] = [min(porta.y3, porta.y4), max(porta.y3, porta.y4)]
        
        if (min_x < track_rev[0][0] < max_x
        and min_y < track_rev[0][1] < max_y):
            for i in range(1, FRAME_PRECEDENTI + 1):
                
                if (track_rev[i] < track_rev[0]
                and (xm, ym) < track_rev[0]
                and min_x < track_rev[i][0] < max_x
                and min_y < track_rev[i][1] < max_y):
                    return (True, porta.numero)
                
    return None


def run_tracker_in_thread(filename, model, file_index):
    passed = None
    scr_width, scr_height = get_screen_size()

    # Store the track history
    track_history = defaultdict(lambda: [])

    cap = cv2.VideoCapture(filename + '.mp4')  # Read the video file
    success, frame = cap.read()

    mask = cv2.imread(filename + '_Mask.png', 0)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = mask // 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out_name = 'track_' + str(filename + '.mp4')
    # out = cv2.VideoWriter(out_name, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    frame_num = 1
    frame_count_pass = 1
    while cap.isOpened() and frame is not None:
        # Read a frame from the video
        roi = frame * mask
        if success:
            # Run YOLOv9 tracking on the frame, persisting tracks between frames
            conf = 0.1
            iou = 0.5
            tracker = "bytetrack_custom.yaml"

            # Esegui l'inferenza
            results = model.track(roi, persist=True, conf=conf, iou=iou, show=False, classes=[0], tracker=tracker)

            for result in results:
                if result.boxes.id is not None:
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        
                        track.append((int(x), int(y)))  # x, y center point
                        if len(track) > 160:  # retain 90 tracks for 160 frames
                            track.pop(0)
                        
                        # Checks if the player has passed through a door 
                        if (len(track) >= FRAME_PRECEDENTI+1): 
                            passed = check(track)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0, 1), thickness=5,
                                      lineType=cv2.LINE_AA)
                    maschera = annotated_frame > 1
                    frame[maschera] = annotated_frame[maschera]
                    
            fontsize = 2
            for porta in PORTE_PonteDestra:
                frame = porta.draw(frame)
            
            cv2.putText(frame, 'Frame ' + str(frame_num), (10, frame.shape[0] - (40 * fontsize)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), 3, cv2.LINE_AA)
            
            if passed is not None: 
                cv2.putText(frame, 'Porta ' + str(passed[1]), (frame.shape[1]*3//4 + 10, frame.shape[0] - (40 * fontsize)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 3, cv2.LINE_AA)
                frame_count_pass += 1
                if frame_count_pass >= 3:
                    passed = None
                    frame_count_pass = 1
                
            frame = cv2.resize(frame, (scr_width, scr_height))
            cv2.imshow(out_name, frame,)
            # Write the frame to the output file
            # out.write(frame)
            frame_num += 1
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
        success, frame = cap.read()

    # Release video sources
    cap.release()


# Load the models
model1 = YOLO("yolov9e-seg.pt")
model2 = YOLO("yolov9e-seg.pt")

# Define the video files for the trackers
ponteDestra = 'IstantaneeCamere/2-PonteDestra'
ponteDestraShort = 'IstantaneeCamere/2-PonteDestraShort'

video_file1 = 'cut.mp4'  # Path to video file, 0 for webcam
video_file2 = 'cut.mp4'  # Path to video file, 0 for webcam, 1 for external camera

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(ponteDestraShort, model1, 1), daemon=True)
# tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(ponteDestra, model2, 2), daemon=True)

# Start the tracker threads
tracker_thread1.start()
# tracker_thread2.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
# tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()
