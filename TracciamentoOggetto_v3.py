import os
import threading
import time
from collections import defaultdict
import tkinter as tk
from CoordinatePorte_170724 import *
# from CoordinatePorte_040924 import *
import types

from ultralytics import YOLO
import numpy as np

VIDEO_ROOT = 'Video_Canoa/'
MASK_ROOT = 'IstantaneeCamere/'
RESULT_ROOT = 'Risultati/'
OFFSET = 15
FRAME_PRECEDENTI = 3

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

# Results
# Array di tuple (int, int), dove il primo elemento è numerato da 1 a 60
porte_passate = [(i, 0, (0, 0)) for i in range(1, 61)]
# il primo valore è il numero della porta, il secodo è l'enum che indica se passato o meno,
# mentre la tupla indica la posizione in cui si è rilevato il passaggio

# Creare un lock per sincronizzare l'accesso all'array
porte_passate_lock = threading.Lock()


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


def check_orientation(pos_corrente, pos_precedente, porta: Porta) -> int:
    if porta.color == GREEN:
        match porta.tipo.value:
            case Entrata.ALTO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] & pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.ALTO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] & pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.BASSO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] & pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
            case Entrata.BASSO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] & pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO.value
                return Passato.PASSATO_MALE.value
    else:
        match porta.tipo.value:
            case Entrata.ALTO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] & pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.ALTO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] & pos_corrente[1] >= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.BASSO_SX.value:
                if pos_corrente[0] >= pos_precedente[0] & pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value
            case Entrata.BASSO_DX.value:
                if pos_corrente[0] <= pos_precedente[0] & pos_corrente[1] <= pos_precedente[1]:
                    return Passato.PASSATO_MALE.value
                return Passato.PASSATO.value


def check(track: list[any], array_porte):
    global segno_os
    track_rev = track.copy()
    track_rev.reverse()
    for porta in array_porte:
        (xm, ym) = (porta.x3 + porta.x4) / 2, (porta.y3 + porta.y4) / 2

        if porta.tipo.value == Entrata.ALTO_SX.value:
            segno_os = [-1, -1, 1, 1]
        elif porta.tipo.value == Entrata.ALTO_DX.value:
            segno_os = [1, -1, -1, 1]
        elif porta.tipo.value == Entrata.BASSO_SX.value:
            segno_os = [-1, 1, 1, -1]
        elif porta.tipo.value == Entrata.BASSO_DX.value:
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
                    return (check_orientation(track_rev[0], track_rev[2], porta), porta.numero)

    return Passato.NON_PASSATO.value


def run_tracker_in_thread(filename, model, file_index):
    global array_porte
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

    out_name = str(RESULT_ROOT + filename + '_track.mp4')
    out = cv2.VideoWriter(
        out_name,
        cv2.VideoWriter.fourcc('m','p','4','v'),
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

    frame_num = 1
    frame_count_pass = 1
    while cap.isOpened() and frame is not None:
        print(str(f"thread {file_index} : frame {frame_num}"))
        # Read a frame from the video
        roi = frame * mask
        if success:
            # Run YOLOv9 tracking on the frame, persisting tracks between frames
            conf = 0.1
            iou = 0.5
            tracker = "bytetrack_custom.yaml"

            # Esegui l'inferenza
            results = model.track(roi, persist=True, conf=conf, iou=iou, show=False, classes=[0], tracker=tracker)
            x = 0
            y = 0
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

                        file_track.write(str(f"Frame {frame_num} - ID {track_id}:\t{int(x)},\t{int(y)} \n"))

                        track.append((int(x), int(y)))  # x, y center point
                        # if len(track) > 160:  # retain 90 tracks for 160 frames
                        #    track.pop(0)

                        # Checks if the player has passed through a door 
                        if (len(track) >= FRAME_PRECEDENTI + 1):
                            passed = check(track, array_porte)

                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 0, 0, 1), thickness=5,
                                      lineType=cv2.LINE_AA)
                    maschera = annotated_frame > 1
                    frame[maschera] = annotated_frame[maschera]

            fontsize = 2

            try:
                for porta in array_porte:
                    frame = porta.draw(frame)
            except UnboundLocalError:
                pass

            cv2.putText(frame, 'Frame ' + str(frame_num), (10, frame.shape[0] - (40 * fontsize)),
                        cv2.FONT_HERSHEY_SIMPLEX, fontsize, (255, 255, 255), 3, cv2.LINE_AA)

            if passed is not None and passed[0] == Passato.PASSATO.value:
                # cv2.putText(frame, 'Porta ' + str(passed[1]),
                #             (frame.shape[1] * 3 // 4 + 10, frame.shape[0] - (40 * fontsize)),
                #             cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 255, 0), 3, cv2.LINE_AA)
                frame_count_pass += 1
                with porte_passate_lock:
                    print(f"Thread {file_index}: Modifica array nella posizione {passed[1]} con risultato {passed[0]}")
                    porte_passate[passed[1]] = (porte_passate[passed[1]][0], passed[0], (int(x), int(y)))
                if frame_count_pass >= 3:
                    passed = None
                    frame_count_pass = 1
            elif passed is not None and passed[0] == Passato.PASSATO_MALE.value:
                # cv2.putText(frame, 'Porta ' + str(passed[1]),
                #             (frame.shape[1] * 3 // 4 + 10, frame.shape[0] - (40 * fontsize)),
                #             cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0, 0, 255), 3, cv2.LINE_AA)
                frame_count_pass += 1
                with porte_passate_lock:
                    print(f"Thread {file_index}: Modifica array nella posizione {passed[1]} con risultato {passed[0]}")
                    porte_passate[passed[1]] = (porte_passate[passed[1]][0], passed[0], (int(x), int(y)))
                if frame_count_pass >= 3:
                    passed = None
                    frame_count_pass = 1

            frame = cv2.resize(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            # cv2.imshow(out_name, frame)
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

    # Release video sources
    print(f"Il video {filename}, elaborato dal thread {file_index}, dura {frame_num/cap.get(cv2.CAP_PROP_FPS)} s \n")
    file_track.close()
    out.release()
    cap.release()


# Load the models
model1 = YOLO("yolov9e-seg.pt")
model2 = YOLO("yolov9e-seg.pt")

# Create the tracker threads

# tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(fn.inizio, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(fn.ponteDestraShort, model1, 2), daemon=True)
# tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(fn.ponteSinistra, model1, 3), daemon=True)
# tracker_thread4 = threading.Thread(target=run_tracker_in_thread, args=(fn.balconeDietro, model1, 4), daemon=True)
# tracker_thread5 = threading.Thread(target=run_tracker_in_thread, args=(fn.balconeAvanti, model1, 4), daemon=True)
# tracker_thread6 = threading.Thread(target=run_tracker_in_thread, args=(fn.lungoCanale, model1, 5), daemon=True)
# tracker_thread7 = threading.Thread(target=run_tracker_in_thread, args=(fn.arrivo, model1, 6), daemon=True)

# Start the tracker threads
#tracker_thread1.start()
tracker_thread2.start()
timer2 = time.time()
# tracker_thread3.start()
# tracker_thread4.start()
# tracker_thread5.start()
# tracker_thread6.start()
# tracker_thread7.start()

# Wait for the tracker threads to finish
#racker_thread1.join()

tracker_thread2.join()
timer2 = time.time() - timer2
print(f"il thread 1 ci ha messo {timer2 // 60} minuti e {timer2 % 60} secondi")
# tracker_thread3.join()
# tracker_thread4.join()
# tracker_thread5.join()
# tracker_thread6.join()
# tracker_thread7.join()

porte_file_path = str(RESULT_ROOT + 'porte.txt')

if not os.path.exists(porte_file_path):
    print(f"File '{porte_file_path}' creato.")
else:
    print(f"File '{porte_file_path}' esiste già.")

file_porte = open(porte_file_path, 'w')

for porta in porte_passate:
    if porta[1] == Passato.PASSATO_MALE.value:
        print(f"nella porta {porta[0]} l'atleta è Passato male")
        file_porte.write(str(f"{int(porta[0])}, {int(porta[1])} , {int(porta[2][0])} , {int(porta[2][1])} \n"))
    elif porta[1] == Passato.PASSATO.value:
        print(f"nella porta {porta[0]} l'atleta è Passato correttamente")
        file_porte.write(str(f"{int(porta[0])}, {int(porta[1])} , {int(porta[2][0])} , {int(porta[2][1])} \n"))
    else:
        print(f"nella porta {porta[0]} l'atleta non è Passato")
        file_porte.write(str(f"{int(porta[0])}, {int(porta[1])} , {int(porta[2][0])} , {int(porta[2][1])} \n"))

# Clean up and close windows
file_porte.close()
cv2.destroyAllWindows()
