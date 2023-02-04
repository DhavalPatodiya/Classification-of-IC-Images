# importing the required module
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

def brain_slice_extraction(img, directory):
    # Making directory in Boundaries Folder
    slices_dirpath = os.path.join('Slices', img)
    cluster_dirpath = os.path.join('Clusters', img)
    os.makedirs(slices_dirpath)
    os.makedirs(cluster_dirpath)
    cluster_count = []

    # Getting the Image
    image_name = os.path.join(directory, img + '.png')
    image = cv2.imread(image_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    j = 0
    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        if w == 4 and h == 5:
            box_image = image[y: y + 118, x + 10: x + 118]
            image1 = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(image1) < 20:
                continue
            j = j + 1
            new_image_name = os.path.join(slices_dirpath, str(j) + ".png")

            cv2.imwrite(new_image_name, box_image)

            img_hsv = cv2.cvtColor(box_image, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 50, 70])
            upper_red = np.array([9, 255, 255])
            red_mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            lower_red = np.array([159, 50, 70])
            upper_red = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

            lower_blue = np.array([90, 50, 70])
            upper_blue = np.array([128, 255, 255])
            blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

            mask = red_mask0 + red_mask1 + blue_mask

            output_img = box_image.copy()
            output_img = cv2.bitwise_and(output_img, output_img, mask=mask)
            gray_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

            if cv2.countNonZero(gray_img) < 20:
                cluster_count.append([str(j), 0])
                blank_image = np.zeros(box_image.shape)
                cv2.imwrite(os.path.join(cluster_dirpath, str(j) + ".png"), blank_image)

            else:
                cluster_img = []
                for p in range(gray_img.shape[0]):
                    for q in range(gray_img.shape[1]):
                        if gray_img[p][q] > 0.0:
                            cluster_img.append([p, q])

                dbscan = DBSCAN(eps=3, min_samples=5, metric ='euclidean', algorithm ='auto').fit(cluster_img)
                labels = dbscan.labels_
                clust_img = np.zeros(output_img.shape)

                for i in range(len(labels)):
                    if labels[i] >= 0:
                        clust_img[cluster_img[i][0], cluster_img[i][1]] = (0, 255, 255)

                unique, counts = np.unique(labels, return_counts=True)
                clust_no = len(counts[counts > 135])
                cluster_count.append([str(j), clust_no])
                cv2.imwrite(os.path.join(cluster_dirpath, str(j) + ".png"), clust_img)


        detected_cluster = os.path.join(cluster_dirpath, "detected_cluster.csv")
        df = pd.DataFrame(cluster_count, columns=['SliceNumber', 'ClusterCount'])
        df.to_csv(detected_cluster, index=False)

