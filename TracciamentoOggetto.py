import math
import os
import random
import threading
import time
import tkinter as tk
import types
from collections import defaultdict

import numpy as np
import torch
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

from CoordinatePorte_170724 import *

VIDEO_ROOT = 'Video_Canoa/'
MASK_ROOT = 'IstantaneeCamere/'
RESULT_ROOT = 'Risultati/'
OFFSET = 30
FRAME_PRECEDENTI = 2
MODEL_FN = 'yolov9e-seg.pt'

# Define the video files and the masks for the trackers
fn = types.SimpleNamespace()
fn.inizio = '1-Inizio'
fn.ponteDestra = '2-PonteDestra'
fn.ponteDestraShort = '2-PonteDestraShort'
fn.ponteSinistra = '3-PonteSinistra'
fn.balconeDietro = '4a-BalconeDietro'
fn.balconeAvanti = '4-BalconeAvanti'
fn.lungoCanale = '5-LungoCanale'
fn.arrivo = '6-Arrivo'

porte_passate = [(i, 0, (0, 0)) for i in range(1, 61)]
frame_totali = []
frame_totali_lock = threading.Lock()

porte_passate_lock = threading.Lock()

data = str(input("\nInserisci la data dell'esecuzione in formato dd/mm/aaaa"))
orario = str(input("\nInserisci l'orario dell'esecuzione  in formato hh:mm:ss"))

def is_inside(xp, yp, vertices):
    vertices = [vertices[0], vertices[1], vertices[3], vertices[2]]

    # Crea il poligono usando i vertici
    poligono = Polygon(vertices)

    # Definisci il punto da controllare
    punto = Point(xp, yp)

    # Restituisci True se il punto è all'interno o sul bordo del poligono
    return poligono.intersects(punto)

def check_orientation(pos_corrente, pos_precedente, porta: Porta):
    if porta.color == GREEN:
        match porta.tipo.value:
            case Entrata.ALTO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] and pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.ALTO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] and pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.BASSO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] and pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.BASSO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] and pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
    else:
        match porta.tipo.value:
            case Entrata.ALTO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] and pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.ALTO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] and pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.BASSO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] and pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.BASSO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] and pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value

def define_check_rects(porta):
    if porta.tipo.value == Entrata.ALTO_SX.value:
        segno_os = [0, -1, 4, 1]
    elif porta.tipo.value == Entrata.ALTO_DX.value:
        segno_os = [0, -1, -4, 1]
    elif porta.tipo.value == Entrata.BASSO_SX.value:
        segno_os = [-3, 1, 0, -1]
    elif porta.tipo.value == Entrata.BASSO_DX.value:
        segno_os = [3, 1, 0, -1]

    vertici_full = [
        (porta.x3 + segno_os[0] * OFFSET, porta.y3 + segno_os[1] * OFFSET),
        (porta.x4 + segno_os[0] * OFFSET, porta.y4 + segno_os[1] * OFFSET),
        (porta.x3 + segno_os[2] * OFFSET, porta.y3 + segno_os[3] * OFFSET),
        (porta.x4 + segno_os[2] * OFFSET, porta.y4 + segno_os[3] * OFFSET)
    ]
    dx = porta.x4 - porta.x3
    dy = porta.y4 - porta.y3
    d = math.sqrt(dx ** 2 + dy ** 2)

    # Normalizzazione del vettore direttore
    u_x = dx / d
    u_y = dy / d

    x3 = (porta.x3 + OFFSET * u_x
          if porta.tipo.value == Entrata.ALTO_DX.value or porta.tipo.value == Entrata.ALTO_SX.value
          else porta.x3 - OFFSET * u_x)
    y3 = (porta.y3 + OFFSET * u_y
          if porta.tipo.value == Entrata.ALTO_DX.value or porta.tipo.value == Entrata.ALTO_SX.value
          else porta.y3 - OFFSET * u_y)
    x4 = (porta.x4 + OFFSET * u_x
          if porta.tipo.value == Entrata.ALTO_DX.value or porta.tipo.value == Entrata.ALTO_SX.value
          else porta.x4 - OFFSET * u_x)
    y4 = (porta.y4 + OFFSET * u_y
          if porta.tipo.value == Entrata.ALTO_DX.value or porta.tipo.value == Entrata.ALTO_SX.value
          else porta.y4 - OFFSET * u_y)

    if porta.tipo.value == Entrata.ALTO_DX.value or porta.tipo.value == Entrata.ALTO_SX.value:
        vertici_ax = [vertici_full[0], vertici_full[1], (porta.x3, porta.y3), (porta.x4, porta.y4)]

        vertici_px = [(x3, y3), (x4, y4),
                      (vertici_full[2][0] + u_x, vertici_full[2][1] + u_y),
                      (vertici_full[3][0] + u_x, vertici_full[3][1] + u_y)]

    elif porta.tipo.value == Entrata.BASSO_DX.value or porta.tipo.value == Entrata.BASSO_SX.value:
        vertici_ax = [
            (vertici_full[0][0] + u_x, vertici_full[0][1] + u_y),
            (vertici_full[1][0] + u_x, vertici_full[1][1] + u_y),
            (x3, y3), (x4, y4)]

        vertici_px = [(porta.x3, porta.y3), (porta.x4, porta.y4), vertici_full[2], vertici_full[3]]
    return vertici_ax, vertici_px

def check(track: list[any], array_porte, frame):
    # global segno_os
    track_rev = track.copy()
    track_rev.reverse()
    for porta in array_porte:
        if porte_passate[porta.numero - 1][1] != Passato.NON_PASSATO.value[0]:
            continue

        # (xm, ym) = (porta.x3 + porta.x4) / 2, (porta.y3 + porta.y4) / 2
        # segno_os = [
        # offsetx_alto,
        # offsety_alto,
        # offsetx_basso,
        # offsety_basso
        # ]

        vertici_ax, vertici_px = define_check_rects(porta)

        if is_inside(track_rev[0][0], track_rev[0][1], vertici_px):
            for i in range(1, FRAME_PRECEDENTI + 1):
                if is_inside(track_rev[i][0], track_rev[i][1], vertici_ax):
                    return check_orientation(track_rev[0], track_rev[2], porta)[0], porta.numero

    return None

def run_tracker_in_thread(filename, file_index):
    # Instantiate a separate model object within each thread to ensure they do not share state which could
    # lead to conflicts. This means calling YOLO('yolov8n.pt') inside the run_tracker_in_thread function
    # for each thread, instead of passing a shared model.
    model = YOLO(MODEL_FN)

    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("CUDA device NOT found. Using CPU for inference.")

    passed = None

    # Store the track history
    track_history = defaultdict(lambda: [])
    id_color = defaultdict(list)
    cap = cv2.VideoCapture(VIDEO_ROOT + filename + '.mp4')  # Read the video file
    success, frame = cap.read()
    numero_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mask = cv2.imread(MASK_ROOT + filename + '_Mask.png', 0)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = mask // 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Define the codec and create a VideoWriter object

    out_name = str(RESULT_ROOT + filename + '_track.mp4')
    out = cv2.VideoWriter(
        out_name,
        cv2.VideoWriter.fourcc('m', 'p', '4', 'v'),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    track_file_path = str(RESULT_ROOT + filename + '_track.txt')

    if not os.path.exists(track_file_path):
        print(f"File '{track_file_path}' creato.")

    file_track = open(track_file_path, 'w')
    try:
        match filename:
            case fn.inizio:
                array_porte = PORTE_Inizio
            case fn.ponteDestra | fn.ponteDestraShort:
                array_porte = PORTE_PonteDestra
            case fn.ponteSinistra:
                array_porte = PORTE_PonteSinistra
            case fn.balconeDietro:
                array_porte = PORTE_BalconeDietro
            case fn.balconeAvanti:
                array_porte = PORTE_BalconeAvanti
            case fn.lungoCanale:
                array_porte = PORTE_LungoCanale
            case fn.arrivo:
                array_porte = PORTE_Arrivo
    except NameError:
        pass

    file_track.write("Data: " + data + "Ora:" + orario + "\n")

    frame_num = 1
    frame_count_pass = 1
    pass_print = (0, 0)
    while cap.isOpened() and frame is not None:
        print(str(f"thread {file_index} : frame {frame_num} of {numero_frame}"))
        # Read a frame from the video
        if RESULT_ROOT == 'Risultati/':
            roi = frame * mask
        else:
            roi = frame
        if success:
            # Run YOLOv9 tracking on the frame, persisting tracks between frames
            conf = 0.1
            iou = 0.5
            tracker = "bytetrack_custom.yaml"

            # Esegui l'inferenza
            results = model.track(roi, persist=True, conf=conf, iou=iou, show=False, classes=[0], tracker=tracker)
            x = y = 0
            for result in results:
                if result.boxes.id is not None:
                    # Get the boxes and track IDs
                    boxes = result.boxes.xywh.cpu()
                    track_ids = result.boxes.id.int().cpu().tolist()
                    # Visualize the results on the frame
                    annotated_frame = result.plot()
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]

                        file_track.write(str(f"Frame {frame_num} - ID {track_id}:\t{int(x)},\t{int(y)} \n"))

                        track.append((int(x), int(y)))
                        # if len(track) > 160:  # retain 90 tracks for 160 frames
                        #    track.pop(0)

                        # Checks if the player has passed through a door 
                        if len(track) >= FRAME_PRECEDENTI + 1:
                            passed = check(track, array_porte, frame_num)

                    maschera = annotated_frame > 1
                    frame[maschera] = annotated_frame[maschera]
            unique_ids = list(track_history.keys())
            for id in unique_ids:
                if id not in id_color:
                    id_color[id] = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 125))]
            for id_number, track in track_history.items():
                # Filtra i punti del track per il frame corrente
                points = [(x, y) for x, y in track]
                color = id_color[id_number][0]
                # Se ci sono almeno due punti, disegna la polyline
                if len(points) > 1:
                    points = np.array(points).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
            fontsize = 2

            if passed is not None and passed[0] is not Passato.NON_PASSATO.value[0]:
                frame_count_pass = 1
                pass_print = passed
                with porte_passate_lock:
                    print(
                        f"Thread {file_index}: Modifica arrayPorte nella posizione {passed[1]} con risultato {passed[0]}")
                    porte_passate[passed[1] - 1] = (passed[1], passed[0], (int(x), int(y)))

            if 6 >= frame_count_pass > 0:
                if pass_print[0] == Passato.PASSATO.value[0]:
                    cv2.putText(frame, 'Passata ' + str(pass_print[1]),
                                (10, frame.shape[0] - (40 * fontsize)),
                                cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 3, cv2.LINE_AA)
                elif pass_print[0] == Passato.PASSATO_MALE.value[0]:
                    cv2.putText(frame, 'Passata Male ' + str(pass_print[1]),
                                (10, frame.shape[0] - (40 * fontsize)),
                                cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), 3, cv2.LINE_AA)
                frame_count_pass += 1

            frame = cv2.resize(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            cv2.imshow(out_name, frame)
            # Write the frame to the output file
            out.write(frame)
            frame_num += 1
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
        success, frame = cap.read()
    with frame_totali_lock:
        frame_totali.append(frame_num)
    # Release video sources
    print(f"Il video {filename}, elaborato dal thread {file_index}, dura {frame_num / cap.get(cv2.CAP_PROP_FPS)} s \n")
    file_track.close()
    out.release()
    cap.release()


# Create the tracker threads
# tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(fn.inizio, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(fn.ponteDestra, 2), daemon=True)
tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(fn.ponteSinistra, 3), daemon=True)
# tracker_thread4 = threading.Thread(target=run_tracker_in_thread, args=(fn.balconeDietro, 4), daemon=True)
tracker_thread5 = threading.Thread(target=run_tracker_in_thread, args=(fn.balconeAvanti, 4), daemon=True)
tracker_thread6 = threading.Thread(target=run_tracker_in_thread, args=(fn.lungoCanale, 5), daemon=True)
tracker_thread7 = threading.Thread(target=run_tracker_in_thread, args=(fn.arrivo, 6), daemon=True)

frame_rate = 30
# Start the tracker threads
timer = time.time()
# tracker_thread1.start()
# timer1 = time.time()
tracker_thread2.start()
timer2 = time.time()
tracker_thread3.start()
timer3 = time.time()
# tracker_thread4.start()
# timer4 = time.time()
tracker_thread5.start()
timer5 = time.time()
tracker_thread6.start()
timer6 = time.time()
tracker_thread7.start()
timer7 = time.time()

# Wait for the tracker threads to finish
# tracker_thread1.join()
# timer1 = time.time() - timer1
# print(f"il thread 1 ha impiegato {timer1 // 60} minuti e {int(timer1 % 60)} secondi")
tracker_thread2.join()
timer2 = time.time() - timer2
print(f"il thread 2 ha impiegato {timer2 // 60} minuti e {int(timer2 % 60)} secondi")
tracker_thread3.join()
timer3 = time.time() - timer3
print(f"il thread 3 ha impiegato {timer3 // 60} minuti e {int(timer3 % 60)} secondi")
# tracker_thread4.join()
# timer4 = time.time() - timer4
# print(f"il thread 4 ha impiegato {timer4 // 60} minuti e {int(timer4 % 60)} secondi")
tracker_thread5.join()
timer5 = time.time() - timer5
print(f"il thread 5 ha impiegato {timer5 // 60} minuti e {int(timer5 % 60)} secondi")
tracker_thread6.join()
timer6 = time.time() - timer6
print(f"il thread 6 ha impiegato {timer6 // 60} minuti e {int(timer6 % 60)} secondi")
tracker_thread7.join()
timer7 = time.time() - timer7
print(f"il thread 7 ha impiegato {timer7 // 60} minuti e {int(timer7 % 60)} secondi")

timer = time.time() - timer
print(f"l'esecuzione ha impiegato {timer // 60} minuti e {int(timer % 60)} secondi")

porte_file_path = str(RESULT_ROOT + 'porte.txt')

if not os.path.exists(porte_file_path):
    print(f"File '{porte_file_path}' creato.")
else:
    print(f"File '{porte_file_path}' esiste già.")

file_porte = open(porte_file_path, 'w')

secondi = round(sum(frame_totali) / frame_rate, 3)
print(secondi)
file_porte.write("Tempo: " + str(secondi) + "\n")
for porta in porte_passate:
    if porta[1] == Passato.PASSATO_MALE.value[0]:
        print(f"nella porta {porta[0]} l'atleta è Passato male")
        file_porte.write(str(f"{porta[0]},\t{porta[1]},\t{porta[2][0]},\t{porta[2][1]}\n"))
    elif porta[1] == Passato.PASSATO.value[0]:
        print(f"nella porta {porta[0]} l'atleta è Passato correttamente")
        file_porte.write(str(f"{porta[0]},\t{porta[1]},\t{porta[2][0]},\t{porta[2][1]}\n"))
    else:
        print(f"nella porta {porta[0]} l'atleta non è Passato")
        file_porte.write(str(f"{porta[0]},\t{porta[1]},\t{porta[2][0]},\t{porta[2][1]}\n"))

# Clean up and close windows
file_porte.close()
cv2.destroyAllWindows()
