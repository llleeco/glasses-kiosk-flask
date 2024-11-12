import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from skimage import io
from itertools import compress

#피부 색상 추출
class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.IMAGE = img.reshape((img.shape[0] * img.shape[1], 3))

        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(self.IMAGE)

        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_

    def getDominantColor(self):
    # 가장 큰 클러스터의 색상만 반환 (주요 색상)
        num_labels = np.arange(0, self.CLUSTERS+1)
        hist, _ = np.histogram(self.LABELS, bins=num_labels)
        dominant = self.COLORS[hist.argmax()].astype(int)
        return dominant  # 단일 RGB 배열 (e.g., [255, 224, 189])
