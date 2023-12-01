import cv2
import numpy as np
import matplotlib.pyplot as plt

#Paso 1: Construcción del espacio de escalas
def espacio_escalas(imagen, octavas, intervalos):
    espacioDescalas = []
    # Parámetros iniciales
    sigma = 1.6
    k = 2**(1/intervalos)

    #  diferentes escalas y octavas
    for octava in range(octavas):
        for intervalo in range(intervalos+3):
            sigma_actual = sigma * (k**intervalo)
            tam = int(6 * sigma_actual + 1)
            if tam % 2 == 0:
                tam += 1

            gauss = cv2.GaussianBlur(imagen, (tam, tam), sigma_actual)
            espacioDescalas.append(gauss)
        imagen = cv2.resize(imagen, (int(imagen.shape[1] / 2), int(imagen.shape[0] / 2)))

    return espacioDescalas


#Paso 2: Localización de puntos clave mediante las
#diferencias de gaussianas
def difgaussianas(espacioDescalas, octavas, intervalos):
    piramideG = []

    for octava in range(octavas):
        for intervalo in range(intervalos + 2):
            i1 = octava * (intervalos + 3)+intervalo
            i2 = octava * (intervalos + 3)+intervalo+1

            dog = espacioDescalas[i2] - espacioDescalas[i1]
            piramideG.append(dog)

    return piramideG

imagen = cv2.imread('2.jpg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
octavas = 4
intervalos = 2

espacio =espacio_escalas(gris, octavas, intervalos)
piramide= difgaussianas(espacio, octavas, intervalos)
####################################################
fig, axs = plt.subplots(octavas, intervalos + 3, figsize=(15, 10))
for i in range(octavas):
    for j in range(intervalos + 3):
        idx = i * (intervalos + 3) + j
        axs[i, j].imshow(espacio[idx], cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title(f'{i + 1}° octava')
fig.suptitle('Espacio de escalas', fontsize=16)
plt.show()
####################################################
fig1, axsis = plt.subplots(octavas, intervalos + 2, figsize=(15, 10))
for i in range(octavas):
    for j in range(intervalos + 2):
        idx = i * (intervalos + 2) + j
        axsis[i, j].imshow(piramide[idx], cmap='gray')
        axsis[i, j].axis('off')
fig1.suptitle('Diferencias de gaussianas', fontsize=16)
plt.show()


#Paso 3: Sacar la orientación para conseguir la invarianza
def asignar_orientaciones(piramide, octavas, intervalos):
    keypoints = []

    for octava in range(octavas):
        for inter in range(1,intervalos+1):
            idx = octava * (intervalos+2) + inter
            localpuntos= cv2.goodFeaturesToTrack(
                piramide[idx], maxCorners=800, qualityLevel=0.1, minDistance=2)
            #print(f"Número de puntos clave: {len(localpuntos)}")
            for kp in localpuntos:
                x, y = kp.ravel()
                x = x / piramide[idx].shape[1]
                y= y / piramide[idx].shape[0]
                magnitudes = []
                orienta = []

                for dx in range(-1, 2):
                  for dy in range(-1, 2):
                    x_idx = int(x + dx)
                    y_idx = int(y + dy)


                    if 0 <= x_idx < piramide[idx].shape[1] and 0 <= y_idx < piramide[idx].shape[0]:
            # Calcular la magnitud y orientación del gradiente en el punto
                       magnitud = np.sqrt(
                        (piramide[idx][y_idx, x_idx + 1] - piramide[idx][y_idx, x_idx - 1])**2 +
                        (piramide[idx][y_idx + 1, x_idx] - piramide[idx][y_idx - 1, x_idx])**2
                        )
                       orientacion = np.arctan2(
                          piramide[idx][y_idx + 1, x_idx] - piramide[idx][y_idx - 1, x_idx],
                          piramide[idx][y_idx, x_idx + 1] - piramide[idx][y_idx, x_idx - 1]
                        )

                       magnitudes.append(magnitud)
                       orienta.append(orientacion)

                orientacionmejor = orienta[np.argmax(magnitudes)]
                sigma=1.6
                keypoint= cv2.KeyPoint(x, y, size=2 * sigma)
                keypoint.angle = np.degrees(orientacionmejor)
                keypoints.append(keypoint)

    return keypoints

keypo= asignar_orientaciones(piramide, octavas, intervalos)

keyp = [(kp.pt[0], kp.pt[1], kp.angle) for kp in keypo]
histograma_orientaciones = np.histogram([kp[2] for kp in keyp], bins=360, range=(0, 360))[0]
plt.bar(range(360), histograma_orientaciones)
plt.title('Histograma de Orientaciones')
plt.xlabel('Bin de Orientación')
plt.ylabel('Magnitudes')
plt.show()



img1 = cv2.imread('5.jpg')
img2 = cv2.imread('6.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img1=cv2.resize(img1,(300,300))
img2=cv2.resize(img2,(300,300))

sift = cv2.xfeatures2d.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
plt.imshow(img3),plt.show()
