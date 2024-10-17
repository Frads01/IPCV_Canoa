from enum import Enum
import cv2
from cv2 import Mat
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


RED = (0, 0, 255)
GREEN = (0, 255, 0)

PORTE_Inizio: list[Porta] = []

PORTE_PonteDestra: list[Porta] = [
    Porta(x1=1342, y1=592, x2=1491, y2=499, x3=1331, y3=796, x4=1468, y4=762, color=GREEN, numero=30, tipo=Entrata.ALTO_SX),
    Porta(x1=1526, y1=483, x2=1600, y2=462, x3=1517, y3=666, x4=1594, y4=640, color=GREEN, numero=29, tipo=Entrata.ALTO_SX),
    Porta(x1=1219, y1=496, x2=1321, y2=457, x3=1209, y3=676, x4=1316, y4=647, color=GREEN, numero=28, tipo=Entrata.ALTO_SX),
    Porta(x1=658, y1=444, x2=805, y2=444, x3=664, y3=628, x4=805, y4=614, color=RED, numero=27, tipo=Entrata.ALTO_SX),
    Porta(x1=610, y1=451, x2=720, y2=442, x3=614, y3=592, x4=723, y4=581, color=GREEN, numero=26, tipo=Entrata.ALTO_SX),
    Porta(x1=1015, y1=432, x2=1090, y2=398, x3=1015, y3=520, x4=1090, y4=517, color=GREEN, numero=25, tipo=Entrata.ALTO_SX),
    Porta(x1=474, y1=417, x2=576, y2=423, x3=474, y3=543, x4=576, y4=547, color=RED, numero=24, tipo=Entrata.ALTO_SX)
]

PORTE_PonteSinistra: list[Porta] = [
    Porta(x1=1217, y1=264, x2=400, y2=230, x3=1152, y3=1080, x4=413, y4=1080, color=GREEN, numero=31, tipo=Entrata.BASSO_SX),
    Porta(x1=665, y1=119, x2=315, y2=68, x3=653, y3=482, x4=321, y4=459, color=RED, numero=33, tipo=Entrata.BASSO_SX),
    Porta(x1=1355, y1=114, x2=1151, y2=65, x3=1338, y3=355, x4=1132, y4=325, color=GREEN, numero=34, tipo=Entrata.BASSO_DX),
    Porta(x1=497, y1=152, x2=330, y2=159, x3=493, y3=296, x4=328, y4=300, color=GREEN, numero=35, tipo=Entrata.BASSO_DX),
    # Porta(x1=589, y1=133, x2=462, y2=135, x3=585, y3=287, x4=460, y4=293, color=GREEN, numero=36, tipo=Entrata.BASSO_DX),
    Porta(x1=1052, y1=149, x2=944, y2=143, x3=1045, y3=284, x4=940, y4=288, color=GREEN, numero=36, tipo=Entrata.BASSO_DX),
    Porta(x1=1030, y1=135, x2=916, y2=143, x3=1021, y3=266, x4=911, y4=271, color=GREEN, numero=37, tipo=Entrata.BASSO_DX),
    Porta(x1=195, y1=116, x2=100, y2=109, x3=193, y3=254, x4=100, y4=244, color=RED, numero=38, tipo=Entrata.BASSO_DX),
    Porta(x1=640, y1=74, x2=546, y2=92, x3=638, y3=199, x4=540, y4=206, color=GREEN, numero=39, tipo=Entrata.BASSO_DX),
    Porta(x1=448, y1=80, x2=356, y2=69, x3=446, y3=175, x4=355, y4=173, color=RED, numero=40, tipo=Entrata.BASSO_DX),
]

PORTE_BalconeDietro: list[Porta] = [
    Porta(x1=923, y1=381, x2=1030, y2=380, x3=925, y3=547, x4=1031, y4=525, color=GREEN, numero=35, tipo=Entrata.ALTO_SX),
    Porta(x1=1004, y1=411, x2=1110, y2=402, x3=1001, y3=610, x4=1108, y4=590, color=GREEN, numero=36, tipo=Entrata.ALTO_SX),
    Porta(x1=447, y1=539, x2=690, y2=533, x3=470, y3=839, x4=701, y4=807, color=GREEN, numero=37, tipo=Entrata.ALTO_SX),
    Porta(x1=1771, y1=441, x2=1837, y2=421, x3=1762, y3=648, x4=1827, y4=609, color=RED, numero=38, tipo=Entrata.ALTO_SX),
    Porta(x1=1575, y1=478, x2=1685, y2=477, x3=1560, y3=795, x4=1675, y4=751, color=GREEN, numero=39, tipo=Entrata.ALTO_SX),
]


PORTE_BalconeAvanti: list[Porta] = [
    Porta(x1=81, y1=530, x2=34, y2=513, x3=104, y3=743, x4=57, y4=700, color=GREEN, numero=41, tipo=Entrata.BASSO_SX),
    Porta(x1=1497, y1=594, x2=1259, y2=495, x3=1498, y3=807, x4=1262, y4=796, color=GREEN, numero=42, tipo=Entrata.BASSO_SX),
    Porta(x1=1127, y1=437, x2=987, y2=438, x3=1134, y3=652, x4=997, y4=631, color=GREEN, numero=43, tipo=Entrata.BASSO_SX),
    Porta(x1=1563, y1=438, x2=1453, y2=430, x3=1563, y3=575, x4=1456, y4=572, color=GREEN, numero=44, tipo=Entrata.BASSO_SX),
    Porta(x1=1232, y1=441, x2=1133, y2=426, x3=1234, y3=548, x4=1140, y4=535, color=GREEN, numero=45, tipo=Entrata.BASSO_SX),
    Porta(x1=1826, y1=399, x2=1726, y2=413, x3=1826, y3=535, x4=1728, y4=537, color=GREEN, numero=46, tipo=Entrata.BASSO_SX),
    Porta(x1=1452, y1=391, x2=1366, y2=393, x3=1453, y3=499, x4=1368, y4=498, color=RED, numero=47, tipo=Entrata.BASSO_SX),
    Porta(x1=1688, y1=380, x2=1606, y2=376, x3=1690, y3=477, x4=1606, y4=479, color=GREEN, numero=48, tipo=Entrata.BASSO_SX),
    Porta(x1=1830, y1=376, x2=1765, y2=380, x3=1830, y3=465, x4=1764, y4=461, color=RED, numero=49, tipo=Entrata.BASSO_SX),
]


PORTE_LungoCanale: list[Porta] = [
    Porta(x1=122, y1=497, x2=109, y2=459, x3=145, y3=720, x4=129, y4=643, color=RED, numero=50, tipo=Entrata.BASSO_SX),
    Porta(x1=515, y1=300, x2=487, y2=277, x3=510, y3=433, x4=483, y4=407, color=RED, numero=51, tipo=Entrata.BASSO_SX),
    Porta(x1=1007, y1=321, x2=941, y2=303, x3=999, y3=449, x4=931, y4=419, color=RED, numero=52, tipo=Entrata.BASSO_SX),
    Porta(x1=1319, y1=356, x2=1235, y2=343, x3=1301, y3=459, x4=1223, y4=445, color=RED, numero=53, tipo=Entrata.BASSO_SX),
    Porta(x1=1620, y1=399, x2=1533, y2=377, x3=1596, y3=513, x4=1512, y4=491, color=GREEN, numero=54, tipo=Entrata.BASSO_SX),
    Porta(x1=1384, y1=311, x2=1322, y2=299, x3=1371, y3=410, x4=1309, y4=392, color=GREEN, numero=55, tipo=Entrata.BASSO_SX),
]

PORTE_Arrivo: list[Porta] = [
    Porta(x1=500, y1=201, x2=569, y2=194, x3=502, y3=285, x4=570, y4=272, color=RED, numero=52, tipo=Entrata.ALTO_SX),
    Porta(x1=478, y1=229, x2=545, y2=227, x3=482, y3=325, x4=546, y4=317, color=RED, numero=53, tipo=Entrata.ALTO_SX),
    Porta(x1=345, y1=271, x2=426, y2=256, x3=350, y3=398, x4=430, y4=380, color=GREEN, numero=54, tipo=Entrata.ALTO_SX),
    Porta(x1=805, y1=249, x2=869, y2=244, x3=803, y3=373, x4=866, y4=359, color=GREEN, numero=55, tipo=Entrata.ALTO_SX),
]