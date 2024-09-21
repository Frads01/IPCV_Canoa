import threading
from collections import defaultdict
import cv2
import tkinter as tk
from CoordinatePorte_170724 import _170724 as date
from CoordinatePorte_040924 import _040924
from enum import Enum

from cv2 import Mat
from ultralytics import YOLO
import numpy as np


class Entrata(Enum): # Verso di ENTRATA nella porta
    ALTO_SX = 0,
    ALTO_DX = 1,
    BASSO_SX = 2,
    BASSO_DX = 3,
    
class Passato(Enum): 
    NON_PASSATO = 0,
    PASSATO = 1,
    PASSATO_MALE = 2,


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
# Le coordinate fanno riferimento alle porte posizionate in data 4 settembre 2024


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
    
def check_orientation(pos_corrente,pos_precedente, porta: Porta)->int:
    if porta.color == GREEN:
        match porta.tipo.value:
            case Entrata.ALTO_SX.value:
                if pos_corrente[0]>=pos_precedente[0] & pos_corrente[1]>=pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.ALTO_DX.value:
                if pos_corrente[0]<=pos_precedente[0] & pos_corrente[1]>=pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.BASSO_SX.value:
                if pos_corrente[0]>=pos_precedente[0] & pos_corrente[1]<=pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.BASSO_DX.value:
                if pos_corrente[0]<=pos_precedente[0] & pos_corrente[1]<=pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
    else:   
        match porta.tipo.value:
            case Entrata.ALTO_SX.value:
                if pos_corrente[0]>=pos_precedente[0] & pos_corrente[1]>=pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.ALTO_DX.value:
                if pos_corrente[0]<=pos_precedente[0] & pos_corrente[1]>=pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.BASSO_SX.value:
                if pos_corrente[0]>=pos_precedente[0] & pos_corrente[1]<=pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.BASSO_DX.value:
                if pos_corrente[0]<=pos_precedente[0] & pos_corrente[1]<=pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value


def check(track: list[any], array_porte):
    track_rev = track.copy()
    track_rev.reverse()
    for porta in array_porte:
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
                    
                    return (check_orientation(track_rev[0],track_rev[2], porta), porta.numero)
                
    return Passato.NON_PASSATO.value


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
            array_porte = date.PORTE_Inizio
        case '2-PonteDestra' | '2-PonteDestraShort':
            array_porte = date.PORTE_PonteDestra
        case '3-PonteSinistra':
            array_porte = date.PORTE_PonteSinistra
        case '4-BalconeAvanti':
            array_porte = date.PORTE_BalconeAvanti
        case '5-LungoCanale':
            array_porte = date.PORTE_LungoCanale
        case '6-Arrivo':
            array_porte = date.PORTE_Arrivo

                
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
                            passed = check(track, array_porte)

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
            
            
            if passed[0] == Passato.PASSATO.value: 
                cv2.putText(frame, 'Porta ' + str(passed[1]), (frame.shape[1]*3//4 + 10, frame.shape[0] - (40 * fontsize)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 3, cv2.LINE_AA)
                frame_count_pass += 1
                if frame_count_pass >= 3:
                    passed = None
                    frame_count_pass = 1
            elif passed[0] == Passato.PASSATO_MALE.value:
                cv2.putText(frame, 'Porta ' + str(passed[1]), (frame.shape[1]*3//4 + 10, frame.shape[0] - (40 * fontsize)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), 3, cv2.LINE_AA)
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
