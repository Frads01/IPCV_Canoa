import threading
from collections import defaultdict
import cv2
import tkinter as tk
from enum import Enum 

from cv2 import Mat
from ultralytics import YOLO
import numpy as np


class Entrata(Enum): # Verso di ENTRATA nella porta
    ALTO_SX = 0,
    ALTO_DX = 1,
    BASSO_SX = 2,
    BASSO_DX = 3,


class Porta:
    color: (int, int, int) # type: ignore
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

    def __init__(self, x1=0, y1=0, x2=0, y2=0, x3=0, y3=0, x4=0, y4=0, color=(255, 255, 255), numero=0, tipo=Entrata.ALTO_SX):
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

    def new(self, x1, y1, x2, y2, x3, y3, x4, y4, color: tuple, numero=0, tipo: Entrata = Entrata.ALTO_SX):
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


VIDEO_ROOT = 'Video_Canoa/'
MASK_ROOT = 'IstantaneeCamere/'
OFFSET = 15
FRAME_PRECEDENTI = 3
RED = (0, 0, 255)
GREEN = (0, 255, 0)
# Tutte le coordinate sono state prese con immagini di risoluzione 1920x972, con offset di +54px
PORTE_Inizio: list[Porta] = [
    Porta(x1=153, y1=406, x2=185, y2=400, x3=156, y3=459, x4=186, y4=452, color=GREEN, numero=1, tipo=Entrata.ALTO_SX),
    Porta(x1=279, y1=437, x2=308, y2=427, x3=281, y3=480, x4=309, y4=462, color=GREEN, numero=2, tipo=Entrata.ALTO_SX),
    Porta(x1=270, y1=552, x2=310, y2=524, x3=272, y3=616, x4=310, y4=589, color=GREEN, numero=3, tipo=Entrata.ALTO_SX),
    Porta(x1=386, y1=427, x2=412, y2=420, x3=388, y3=471, x4=412, y4=457, color=RED, numero=4, tipo=Entrata.ALTO_SX),
    Porta(x1=451, y1=683, x2=484, y2=638, x3=453, y3=750, x4=483, y4=698, color=GREEN, numero=5, tipo=Entrata.ALTO_SX),
    Porta(x1=680, y1=525, x2=678, y2=493, x3=677, y3=567, x4=676, y4=540, color=RED, numero=6, tipo=Entrata.BASSO_SX), 
    Porta(x1=889, y1=622, x2=857, y2=588, x3=883, y3=674, x4=855, y4=637, color=GREEN, numero=6, tipo=Entrata.BASSO_SX), 
    Porta(x1=1089, y1=580, x2=1064, y2=548, x3=1085, y3=621, x4=1059, y4=594, color=GREEN, numero=7, tipo=Entrata.BASSO_SX),
    Porta(x1=1114, y1=510, x2=1093, y2=497, x3=1109, y3=554, x4=1090, y4=533, color=RED, numero=8, tipo=Entrata.BASSO_SX),
    Porta(x1=1253, y1=593, x2=1218, y2=561, x3=1247, y3=626, x4=1212, y4=608, color=GREEN, numero=9, tipo=Entrata.BASSO_SX),
    Porta(x1=1208, y1=518, x2=1182, y2=502, x3=1203, y3=560, x4=1178, y4=536, color=RED, numero=10, tipo=Entrata.BASSO_SX),
    Porta(x1=1221, y1=460, x2=1205, y2=457, x3=1217, y3=500, x4=1203, y4=488, color=RED, numero=11, tipo=Entrata.BASSO_SX),
    Porta(x1=1342, y1=498, x2=1320, y2=480, x3=1337, y3=532, x4=1315, y4=522, color=GREEN, numero=12, tipo=Entrata.BASSO_SX),
    Porta(x1=1400, y1=532, x2=1371, y2=527, x3=1396, y3=566, x4=1367, y4=560, color=GREEN, numero=13, tipo=Entrata.BASSO_SX),
    Porta(x1=1448, y1=471, x2=1426, y2=461, x3=1444, y3=505, x4=1424, y4=497, color=GREEN, numero=14, tipo=Entrata.BASSO_SX),
    Porta(x1=1459, y1=442, x2=1437, y2=436, x3=1456, y3=437, x4=1434, y4=463, color=GREEN, numero=15, tipo=Entrata.BASSO_SX),
    Porta(x1=1498, y1=427, x2=1478, y2=422, x3=1496, y3=452, x4=1476, y4=444, color=GREEN, numero=16, tipo=Entrata.BASSO_SX)
]
PORTE_PonteDestra: list[Porta] = [
    Porta(x1=1342, y1=592, x2=1491, y2=499, x3=1331, y3=796, x4=1468, y4=762, color=GREEN, numero=30, tipo=Entrata.ALTO_SX),
    Porta(x1=1526, y1=483, x2=1600, y2=462, x3=1517, y3=666, x4=1594, y4=640, color=GREEN, numero=29, tipo=Entrata.ALTO_SX),
    Porta(x1=1219, y1=496, x2=1321, y2=457, x3=1209, y3=676, x4=1316, y4=647, color=GREEN, numero=28, tipo=Entrata.ALTO_SX),
    Porta(x1=658, y1=444, x2=805, y2=444, x3=664, y3=628, x4=805, y4=614, color=RED, numero=27, tipo=Entrata.ALTO_SX),
    Porta(x1=610, y1=451, x2=720, y2=442, x3=614, y3=592, x4=723, y4=581, color=GREEN, numero=26, tipo=Entrata.ALTO_SX),
    Porta(x1=1015, y1=432, x2=1090, y2=398, x3=1015, y3=577, x4=1090, y4=517, color=GREEN, numero=25, tipo=Entrata.ALTO_SX),
    Porta(x1=474, y1=417, x2=576, y2=423, x3=474, y3=543, x4=576, y4=547, color=RED, numero=24, tipo=Entrata.ALTO_SX)
]
PORTE_PonteSinistra: list[Porta] = [
    Porta(x1=1170, y1=193, x2=519, y2=197, x3=1139, y3=875, x4=575, y4=957, color=GREEN, numero=31, tipo=Entrata.BASSO_SX),
    Porta(x1=935, y1=100, x2=1239, y2=199, x3=1226, y3=417, x4=941, y4=453, color=RED, numero=33, tipo=Entrata.BASSO_SX),
    Porta(x1=1596, y1=93, x2=1406, y2=73, x3=1580, y3=316, x4=1398, y4=317, color=GREEN, numero=34, tipo=Entrata.BASSO_SX),
    Porta(x1=508, y1=146, x2=343, y2=168, x3=511, y3=300, x4=350, y4=318, color=GREEN, numero=35, tipo=Entrata.BASSO_SX),
    Porta(x1=1087, y1=110, x2=981, y2=123, x3=1086, y3=258, x4=982, y4=271, color=GREEN, numero=36, tipo=Entrata.BASSO_SX),
    Porta(x1=971, y1=112, x2=855, y2=118, x3=971, y3=241, x4=858, y4=248, color=GREEN, numero=37, tipo=Entrata.BASSO_SX),
    Porta(x1=667, y1=113, x2=573, y2=118, x3=667, y3=250, x4=577, y4=250, color=RED, numero=38, tipo=Entrata.BASSO_SX),
    Porta(x1=261, y1=121, x2=154, y2=127, x3=265, y3=248, x4=158, y4=250, color=RED, numero=39, tipo=Entrata.BASSO_SX),
    Porta(x1=743, y1=78, x2=658, y2=75, x3=745, y3=172, x4=657, y4=173, color=GREEN, numero=40, tipo=Entrata.BASSO_SX)
]
# PORTE_BalconeDietro: list[Porta] = []
PORTE_BalconeAvanti: list[Porta] = [# SCALATO
    Porta(x1=403, y1=532, x2=303, y2=531, x3=428, y3=767, x4=322, y4=742, color=GREEN, numero=41, tipo=Entrata.BASSO_SX),
    Porta(x1=789, y1=449, x2=642, y2=448, x3=816, y3=731, x4=662, y4=722, color=GREEN, numero=42, tipo=Entrata.BASSO_SX),
    Porta(x1=1035, y1=439, x2=910, y2=437, x3=1045, y3=632, x4=918, y4=614, color=GREEN, numero=43, tipo=Entrata.BASSO_SX),
    Porta(x1=1354, y1=424, x2=1238, y2=407, x3=1358, y3=582, x4=1242, y4=570, color=GREEN, numero=44, tipo=Entrata.BASSO_SX),
    Porta(x1=1784, y1=434, x2=1663, y2=426, x3=1787, y3=560, x4=1666, y4=569, color=RED, numero=45, tipo=Entrata.BASSO_SX),
    Porta(x1=1687, y1=406, x2=1594, y2=411, x3=1686, y3=534, x4=1594, y4=530, color=RED, numero=46, tipo=Entrata.BASSO_SX),
    Porta(x1=1497, y1=397, x2=1411, y2=401, x3=1502, y3=509, x4=1414, y4=504, color=GREEN, numero=47, tipo=Entrata.BASSO_SX)
]
PORTE_LungoCanale: list[Porta] = [# SCALATO ????
    Porta(x1=382, y1=417, x2=367, y2=316, x3=386, y3=503, x4=365, y4=463, color=GREEN, numero=50, tipo=Entrata.BASSO_SX),
    Porta(x1=866, y1=456, x2=796, y2=410, x3=853, y3=651, x4=784, y4=600, color=RED, numero=51, tipo=Entrata.BASSO_SX),
    Porta(x1=1203, y1=396, x2=1122, y2=355, x3=1186, y3=524, x4=1111, y4=484, color=GREEN, numero=52, tipo=Entrata.BASSO_SX),
    Porta(x1=1291, y1=294, x2=1234, y2=283, x3=1278, y3=388, x4=1222, y4=373, color=RED, numero=53, tipo=Entrata.BASSO_SX),
    Porta(x1=1529, y1=384, x2=1444, y2=356, x3=1514, y3=494, x4=1430, y4=465, color=GREEN, numero=54, tipo=Entrata.BASSO_SX),
    Porta(x1=1604, y1=361, x2=1554, y2=342, x3=1586, y3=454, x4=1538, y4=432, color=GREEN, numero=55, tipo=Entrata.BASSO_SX)
]
PORTE_Arrivo: list[Porta] = [# SCALATO
    Porta(x1=361, y1=229, x2=436, y2=214, x3=364, y3=308, x4=437, y4=303, color=RED, numero=52, tipo=Entrata.ALTO_SX),
    Porta(x1=429, y1=266, x2=504, y2=243, x3=429, y3=388, x4=504, y4=364, color=GREEN, numero=54, tipo=Entrata.ALTO_SX),
    Porta(x1=869, y1=238, x2=922, y2=232, x3=862, y3=353, x4=917, y4=338, color=GREEN, numero=55, tipo=Entrata.ALTO_SX),
    Porta(x1=873, y1=359, x2=981, y2=344, x3=866, y3=520, x4=965, y4=499, color=RED, numero=56, tipo=Entrata.ALTO_SX),
    Porta(x1=1275, y1=511, x2=1355, y2=481, x3=1248, y3=671, x4=1334, y4=624, color=GREEN, numero=57, tipo=Entrata.ALTO_SX),
    Porta(x1=1201, y1=589, x2=1329, y2=548, x3=1175, y3=809, x4=1290, y4=760, color=GREEN, numero=58, tipo=Entrata.ALTO_SX),
    Porta(x1=1721, y1=476, x2=1750, y2=447, x3=1694, y3=604, x4=1727, y4=568, color=GREEN, numero=59, tipo=Entrata.ALTO_SX),
    Porta(x1=1718, y1=689, x2=1747, y2=646, x3=1652, y3=888, x4=1697, y4=829, color=GREEN, numero=60, tipo=Entrata.ALTO_SX)
]


def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return width, height


def cross_product(xa, ya, xb, yb):
    return xa * yb - ya * xb


def is_point_in_rotated_rectangle(xp, yp, vertices):
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    x3, y3 = vertices[2]
    x4, y4 = vertices[3]
    
    cp1 = cross_product(x2 - x1, y2 - y1, xp - x1, yp - y1)
    cp2 = cross_product(x3 - x2, y3 - y2, xp - x2, yp - y2)
    cp3 = cross_product(x4 - x3, y4 - y3, xp - x3, yp - y3)
    cp4 = cross_product(x1 - x4, y1 - y4, xp - x4, yp - y4)

    if (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0):
        return True
    else:
        return False


def check(track: list[any]):
    track_rev = track.copy()
    track_rev.reverse()
    for porta in PORTE_PonteDestra:
        (xm, ym) = (porta.x3+porta.x4)/2, (porta.y3+porta.y4)/2
        
        if porta.tipo == Entrata.ALTO_SX:
            segno_os = [-1, -1, 1, 1]
        elif porta.tipo == Entrata.ALTO_DX:
            segno_os = [1, -1, -1, 1] 
        elif porta.tipo == Entrata.BASSO_SX:
            segno_os = [-1, 1, 1, -1]     
        elif porta.tipo == Entrata.BASSO_DX:
            segno_os = [1, 1, -1, -1]
        
        vertici_full = [
            (porta.x3 + segno_os[0] * OFFSET, porta.y3 + segno_os[1] * OFFSET),
            (porta.x4 + segno_os[0] * OFFSET, porta.y3 + segno_os[1] * OFFSET),
            (porta.x3 + segno_os[2] * OFFSET, porta.y3 + segno_os[3] * OFFSET),
            (porta.x4 + segno_os[2] * OFFSET, porta.y4 + segno_os[3] * OFFSET)
        ]
        vertici_ax = [vertici_full[0], vertici_full[1], (porta.x3, porta.y3), (porta.x4, porta.y4)]
        vertici_px = [(porta.x3, porta.y3), (porta.x4, porta.y4), vertici_full[2], vertici_full[3]]
        
        if (is_point_in_rotated_rectangle(track_rev[0][0], track_rev[0][1], vertici_px)):
            for i in range(1, FRAME_PRECEDENTI + 1):
                if (is_point_in_rotated_rectangle(track_rev[i][0], track_rev[i][1], vertici_ax)):
                    return (True, porta.numero)
                
    return None


def run_tracker_in_thread(filename, model, file_index):
    passed = None
    scr_width, scr_height = get_screen_size()

    # Store the track history
    track_history = defaultdict(lambda: [])

    cap = cv2.VideoCapture(VIDEO_ROOT + filename + '.mp4')  # Read the video file
    success, frame = cap.read()

    mask = cv2.imread(MASK_ROOT + filename + '_Mask.png', 0)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = mask // 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out_name = 'track_' + str(filename + '.mp4')
    
    match filename:
        case '1-Inizio':
            array_porte = PORTE_Inizio
        case '2-PonteDestra':
            array_porte = PORTE_PonteDestra
        case '3-PonteSinistra':
            array_porte = PORTE_PonteSinistra
        case '4-BalconeAvanti':
            array_porte = PORTE_BalconeAvanti
        case '5-LungoCanale':
            array_porte = PORTE_LungoCanale
        case '6-Arrivo':
            array_porte = PORTE_Arrivo
        case '2-PonteDestraShort': 
            array_porte = PORTE_PonteDestra
            
    
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
            for porta in array_porte:
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

# Define the video files and the masks for the trackers
inizio = '1-Inizio'
ponteDestra = '2-PonteDestra'
ponteDestraShort = '2-PonteDestraShort'
ponteSinistra = '3-PonteSinistra'
balconeAvanti = '4-BalconeAvanti'
lungoCanale = '5-LungoCanale'
arrivo = '6-Arrivo'

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(ponteDestra, model1, 1), daemon=True)
# tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(ponteDestra, model1, 2), daemon=True)
# tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(ponteSinistra, model1, 3), daemon=True)
# tracker_thread4 = threading.Thread(target=run_tracker_in_thread, args=(balconeDietro, model1, 4), daemon=True)
# tracker_thread5 = threading.Thread(target=run_tracker_in_thread, args=(balconeAvanti, model1, 4), daemon=True)
# tracker_thread6 = threading.Thread(target=run_tracker_in_thread, args=(lungoCanale, model1, 5), daemon=True)
# tracker_thread7 = threading.Thread(target=run_tracker_in_thread, args=(arrivo, model1, 6), daemon=True)

# Start the tracker threads
tracker_thread1.start()
# tracker_thread2.start()
# tracker_thread3.start()
# tracker_thread4.start()
# tracker_thread5.start()
# tracker_thread6.start()
# tracker_thread7.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
# tracker_thread2.join()
# tracker_thread3.join()
# tracker_thread4.join()
# tracker_thread5.join()
# tracker_thread6.join()
# tracker_thread7.join()

# Clean up and close windows
cv2.destroyAllWindows()
