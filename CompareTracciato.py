import math
import os
import threading
import time
from collections import defaultdict
import tkinter as tk
import random

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
        # Inizializza un dizionario per memorizzare le tracce di ogni ID
        tracks_per_id = defaultdict(list)
        #SALTARE PRIMA LINEA, NEL MOMENTO IN CUI SARà PRESENTE IL CONTEGGIO DEL TEMPO
        for line in file:
            due_punti  = 0
            virgola = 0
            id = 0
            frame_pos = 6
            trattino=0
            i = 0
            for char in line:
                i += 1
                if char == ":":
                    due_punti = i
                if char == ",":
                    virgola = i
                if char == "-":
                    trattino=i
            x = int(line[due_punti:virgola-1])
            y = int(line[virgola:i-1])
            id = int(line[trattino+4:due_punti-1])
            frame_number = int(line[frame_pos:trattino-2])
            tracks_per_id[id].append((frame_number, x, y))
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
            # Estrai tutti gli ID da tracks_per_id e generali colori per ogni ID unico
            unique_ids = list(tracks_per_id.keys())
            colors = {id_number: (random.randint(0, 125), random.randint(0, 255), random.randint(0, 255)) for id_number in unique_ids}
            while cap.isOpened():
                ret, frame = cap.read()
                # Controlla se il frame è valido
                if not ret or frame is None:
                    break
                frame = cv2.resize(frame, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                
                for id_number, track in tracks_per_id.items():
                    # Filtra i punti del track per il frame corrente
                    points = [(x, y) for f, x, y in track if f <= frame_num]
                    
                    # Se ci sono almeno due punti, disegna la polyline
                    if len(points) > 1:
                        points = np.array(points).astype(np.int32).reshape((-1, 1, 2))
                        color = colors[id_number]  # Ottieni il colore per questo ID
                        cv2.polylines(frame, [points], isClosed=False, color=color, thickness=6, lineType=cv2.LINE_AA)
                        
                # Visualizza il frame con le polylines
                cv2.imshow("Compare", frame)

                frame_num += 1
                # Aggiungi il waitKey per gestire correttamente il ciclo degli eventi
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Premendo 'q' l'utente può chiudere il video
                    break
            cap.release()
while(1):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break