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


PORTE_Inizio: list[Porta] = [
    Porta(x1=153, y1=406, x2=185, y2=401, x3=156, y3=460, x4=186, y4=452, color=GREEN, numero=1, tipo=Entrata.ALTO_SX),
    Porta(x1=279, y1=438, x2=308, y2=427, x3=281, y3=482, x4=309, y4=467, color=GREEN, numero=2, tipo=Entrata.ALTO_SX),
    Porta(x1=270, y1=553, x2=310, y2=526, x3=272, y3=616, x4=310, y4=587, color=GREEN, numero=3, tipo=Entrata.ALTO_SX),
    Porta(x1=386, y1=427, x2=412, y2=419, x3=388, y3=471, x4=412, y4=458, color=RED, numero=4, tipo=Entrata.ALTO_SX),
    Porta(x1=451, y1=683, x2=484, y2=639, x3=453, y3=748, x4=483, y4=700, color=GREEN, numero=5, tipo=Entrata.ALTO_SX),
    Porta(x1=680, y1=523, x2=678, y2=493, x3=677, y3=567, x4=676, y4=540, color=RED, numero=6, tipo=Entrata.BASSO_SX),
    Porta(x1=889, y1=622, x2=857, y2=590, x3=883, y3=672, x4=855, y4=637, color=GREEN, numero=6, tipo=Entrata.BASSO_SX),
    Porta(x1=1089, y1=579, x2=1064, y2=551, x3=1085, y3=621, x4=1059, y4=594, color=GREEN, numero=7, tipo=Entrata.BASSO_SX),
    Porta(x1=1114, y1=511, x2=1093, y2=494, x3=1109, y3=554, x4=1090, y4=537, color=RED, numero=8, tipo=Entrata.BASSO_SX),
    Porta(x1=1253, y1=592, x2=1218, y2=560, x3=1247, y3=626, x4=1212, y4=609, color=GREEN, numero=9, tipo=Entrata.BASSO_SX),
    Porta(x1=1208, y1=520, x2=1182, y2=501, x3=1203, y3=559, x4=1178, y4=539, color=RED, numero=10, tipo=Entrata.BASSO_SX),
    Porta(x1=1221, y1=461, x2=1205, y2=458, x3=1217, y3=502, x4=1203, y4=489, color=RED, numero=11, tipo=Entrata.BASSO_SX),
    Porta(x1=1342, y1=496, x2=1320, y2=482, x3=1337, y3=533, x4=1315, y4=521, color=GREEN, numero=12, tipo=Entrata.BASSO_SX),
    Porta(x1=1400, y1=530, x2=1371, y2=528, x3=1396, y3=569, x4=1367, y4=560, color=GREEN, numero=13, tipo=Entrata.BASSO_SX),
    Porta(x1=1448, y1=471, x2=1426, y2=462, x3=1444, y3=506, x4=1424, y4=497, color=GREEN, numero=14, tipo=Entrata.BASSO_SX),
    Porta(x1=1459, y1=441, x2=1437, y2=434, x3=1456, y3=437, x4=1434, y4=463, color=GREEN, numero=15, tipo=Entrata.BASSO_SX),
    Porta(x1=1498, y1=427, x2=1478, y2=421, x3=1496, y3=453, x4=1476, y4=444, color=GREEN, numero=16, tipo=Entrata.BASSO_SX)
]

PORTE_PonteDestra: list[Porta] = [ #17/07/24
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

PORTE_BalconeDietro: list[Porta] = []

PORTE_BalconeAvanti: list[Porta] = [
    Porta(x1=403, y1=532, x2=303, y2=531, x3=428, y3=767, x4=322, y4=742, color=GREEN, numero=41, tipo=Entrata.BASSO_SX),
    Porta(x1=789, y1=449, x2=642, y2=448, x3=816, y3=731, x4=662, y4=722, color=GREEN, numero=42, tipo=Entrata.BASSO_SX),
    Porta(x1=1035, y1=439, x2=910, y2=437, x3=1045, y3=632, x4=918, y4=614, color=GREEN, numero=43, tipo=Entrata.BASSO_SX),
    Porta(x1=1354, y1=424, x2=1238, y2=407, x3=1358, y3=582, x4=1242, y4=570, color=GREEN, numero=44, tipo=Entrata.BASSO_SX),
    Porta(x1=1784, y1=434, x2=1663, y2=426, x3=1787, y3=560, x4=1666, y4=569, color=RED, numero=45, tipo=Entrata.BASSO_SX),
    Porta(x1=1687, y1=406, x2=1594, y2=411, x3=1686, y3=534, x4=1594, y4=530, color=RED, numero=46, tipo=Entrata.BASSO_SX),
    Porta(x1=1497, y1=397, x2=1411, y2=401, x3=1502, y3=509, x4=1414, y4=504, color=GREEN, numero=47, tipo=Entrata.BASSO_SX)
]

PORTE_LungoCanale: list[Porta] = [
    Porta(x1=382, y1=418, x2=367, y2=317, x3=386, y3=503, x4=365, y4=463, color=GREEN, numero=50, tipo=Entrata.BASSO_SX),
    Porta(x1=866, y1=457, x2=796, y2=410, x3=853, y3=651, x4=784, y4=600, color=RED, numero=51, tipo=Entrata.BASSO_SX),
    Porta(x1=1203, y1=397, x2=1122, y2=356, x3=1186, y3=524, x4=1111, y4=484, color=GREEN, numero=52, tipo=Entrata.BASSO_SX),
    Porta(x1=1291, y1=294, x2=1234, y2=283, x3=1278, y3=389, x4=1222, y4=373, color=RED, numero=53, tipo=Entrata.BASSO_SX),
    Porta(x1=1529, y1=384, x2=1444, y2=357, x3=1514, y3=494, x4=1430, y4=466, color=GREEN, numero=54, tipo=Entrata.BASSO_SX),
    Porta(x1=1604, y1=361, x2=1554, y2=342, x3=1586, y3=454, x4=1538, y4=432, color=GREEN, numero=55, tipo=Entrata.BASSO_SX)
]

PORTE_Arrivo: list[Porta] = [
    Porta(x1=361, y1=229, x2=436, y2=214, x3=364, y3=308, x4=437, y4=303, color=RED, numero=52, tipo=Entrata.ALTO_SX),
    Porta(x1=429, y1=266, x2=504, y2=243, x3=429, y3=388, x4=504, y4=364, color=GREEN, numero=54, tipo=Entrata.ALTO_SX),
    Porta(x1=869, y1=238, x2=922, y2=232, x3=862, y3=353, x4=917, y4=338, color=GREEN, numero=55, tipo=Entrata.ALTO_SX),
    Porta(x1=873, y1=359, x2=981, y2=344, x3=866, y3=520, x4=965, y4=499, color=RED, numero=56, tipo=Entrata.ALTO_SX),
    Porta(x1=1275, y1=511, x2=1355, y2=481, x3=1248, y3=671, x4=1334, y4=624, color=GREEN, numero=57, tipo=Entrata.ALTO_SX),
    Porta(x1=1201, y1=589, x2=1329, y2=548, x3=1175, y3=809, x4=1290, y4=760, color=GREEN, numero=58, tipo=Entrata.ALTO_SX),
    Porta(x1=1721, y1=476, x2=1750, y2=447, x3=1694, y3=604, x4=1727, y4=568, color=GREEN, numero=59, tipo=Entrata.ALTO_SX),
    Porta(x1=1718, y1=689, x2=1747, y2=646, x3=1652, y3=888, x4=1697, y4=829, color=GREEN, numero=60, tipo=Entrata.ALTO_SX)
]

