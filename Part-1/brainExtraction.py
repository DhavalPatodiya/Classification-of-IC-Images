# importing the required module
import os
import cv2

def brain_slice_extraction(img, directory):
    # Making directory in Boundaries Folder
    dirpath = os.path.join('Slices', img)
    os.makedirs(dirpath)

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
            new_image_name = os.path.join(dirpath, str(j) + ".png")
            cv2.imwrite(new_image_name, box_image)


def brain_boundary_extraction(item):
    # Making boundaries directory
    dirpath = os.path.join('Boundaries', item)
    os.makedirs(dirpath)

    slices_dir = os.path.join('Slices', item)
    j = 0
    for filename in os.listdir(slices_dir):
        if filename.endswith(".png"):
            image = cv2.imread(os.path.join(slices_dir, filename))
            gray_image = cv2.imread(os.path.join(slices_dir, filename), 0)
            contours, hierachy = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
            j = j + 1
            new_image_name = os.path.join(dirpath, str(j) + ".png")

            cv2.imwrite(new_image_name, image[:, :, ::-1])
    print("length ", j)
