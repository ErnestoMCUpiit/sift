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
