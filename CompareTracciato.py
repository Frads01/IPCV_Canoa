import math
import os
import threading
import time
from collections import defaultdict
import tkinter as tk

import cv2
import numpy as np
from Cython import returns
from scipy.optimize import brent
from sympy.codegen.ast import continue_, break_

from CoordinatePorte_170724 import *
import types

VIDEO_ROOT = 'Video_Canoa/'
MASK_ROOT = 'IstantaneeCamere/'
RESULT_ROOT = 'Risultati/'
# RESULT_ROOT = 'Risultati_NOroi/'

file_nella_cartella = [f for f in os.listdir(RESULT_ROOT) if f.endswith('.txt')
                       and os.path.isfile(os.path.join(RESULT_ROOT, f))]

percorso_file = ""
if not file_nella_cartella:
    print("Non ci sono file nella cartella.")
else:
    # Stampa l'elenco dei file con un indice
    print("Seleziona un file da aprire:")
    for i, file in enumerate(file_nella_cartella):
        print(f"{i + 1}. {file}")

    # Ottieni la scelta dell'utente

    # Controlla che la scelta sia valida
    scelta = -1
    while scelta == -1:
        scelta = int(input("\nInserisci il numero del file che desideri aprire o clicca 'q' per annullare: ")) - 1

        if 0 <= scelta < len(file_nella_cartella):
            file_scelto = file_nella_cartella[scelta]
            percorso_file = os.path.join(RESULT_ROOT, file_scelto)
        elif cv2.waitKey(1) & 0xFF == ord("q"):
            "Annulla"
            break
        else:
            print("Scelta non valida.")

    if percorso_file != "":
        file = open(percorso_file, "r")
        track = []
        for line in file:
            due_punti  = 0
            virgola = 0
            i = 0
            for char in line:
                i += 1
                if char == ":":
                    due_punti = i
                if char == ",":
                    virgola = i
            x = int(line[due_punti:virgola-1])
            y = int(line[virgola:i-1])
            track.append((x,y))
        file.close()
        print("Seleziona ora il video su cui disegnare la traccia da confontare")
        file_nella_cartella = [f for f in os.listdir(RESULT_ROOT) if f.endswith('.mp4')
                               and os.path.isfile(os.path.join(RESULT_ROOT, f))]
        percorso_file = ""
        if not file_nella_cartella:
            print("Non ci sono file nella cartella.")
        else:
            print("Seleziona un file da aprire:")
            for i, file in enumerate(file_nella_cartella):
                print(f"{i + 1}. {file}")
            scelta = -1
            while scelta == -1:
                scelta = int(input("\nInserisci il numero del file che desideri aprire o clicca 'q' per annullare: ")) - 1

                if 0 <= scelta < len(file_nella_cartella):
                    file_scelto = file_nella_cartella[scelta]
                    percorso_file = os.path.join(RESULT_ROOT, file_scelto)
                    break
                else:
                    print("Scelta non valida.")
            cap = cv2.VideoCapture(percorso_file)
            frame_num = 1
            while cap.isOpened():
                ret, frame = cap.read()
                points = np.hstack(track[0:frame_num]).astype(np.int32).reshape((-1, 1, 2))
                frame_num += 1
                cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=6,
                              lineType=cv2.LINE_AA)
                cv2.imshow("Compare", frame)
                if len(track) < frame_num:
                    break
            cap.release()