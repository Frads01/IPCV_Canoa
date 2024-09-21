from TracciamentoOggetto_v3 import Porta, Entrata, GREEN, RED

PORTE_Inizio: list[Porta] = []

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

PORTE_BalconeDietro: list[Porta] = [
  # @TODO
]

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