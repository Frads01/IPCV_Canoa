import math
import os
import threading
import time
from collections import defaultdict
import tkinter as tk
from CoordinatePorte_170724 import *
import types

VIDEO_ROOT = 'Video_Canoa/'
MASK_ROOT = 'IstantaneeCamere/'
RESULT_ROOT = 'Risultati/'
RESULT_ROOT = 'Risultati_NOroi/'

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
    scelta = int(input("\nInserisci il numero del file che desideri aprire o clicca 'q' per annullare: ")) - 1

    # Controlla che la scelta sia valida
    if 0 <= scelta < len(file_nella_cartella):
        file_scelto = file_nella_cartella[scelta]
        percorso_file = os.path.join(RESULT_ROOT, file_scelto)
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        "Annulla"
    else:
        print("Scelta non valida.")

    if percorso_file != "":
        file = open(percorso_file, "r")
        for line in file:
            print(line)
            due_punti  = 0
            virgola = 0
            i = 0
            for char in line:
                i += 1
                if char == ":":
                    due_punti = i
                if char == ",":
                    virgola = i
            x = int(line[due_punti+1:virgola])
            y = int(line[virgola+1:i])