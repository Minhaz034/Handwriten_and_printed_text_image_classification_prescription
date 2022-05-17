import os
import tqdm
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as T
from scipy.ndimage.filters import gaussian_filter


REBUILD_DATA = True
KERNEL = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
handwritten = "./data/handwritten"
printed = "./data/printed"


# augmenting handwritten text:
shapes = []
pseudo_count = 0
for f in tqdm.tqdm(os.listdir(handwritten)):
    aug_1 = "1"+f
    aug_2 = "2"+f
    aug_3 = "3"+f
    aug_4 = "4"+f
    aug_5 = "5"+f
    aug_6 = "6"+f

    img = cv.imread(os.path.join(handwritten,f), cv.IMREAD_GRAYSCALE)

    # img = cv.resize(img,[W,H],interpolation=cv.INTER_AREA)
    cv.imwrite("data/handwritten-augmented/"+aug_1,img)
    img_blur = gaussian_filter(img, sigma=3)
    cv.imwrite("data/handwritten-augmented/"+aug_2,img_blur)
    img_sh = cv.filter2D(src=img, ddepth=-1, kernel=KERNEL)
    cv.imwrite("data/handwritten-augmented/"+aug_3,img_sh)

    flipr = np.fliplr(img)
    cv.imwrite("data/handwritten-augmented/"+aug_4,flipr)
    flip_blur = gaussian_filter(flipr, sigma=3)
    cv.imwrite("data/handwritten-augmented/"+aug_5,flip_blur)
    flip_sh = cv.filter2D(src=flipr, ddepth=-1, kernel=KERNEL)
    cv.imwrite("data/handwritten-augmented/"+aug_6,flip_sh)

    # plt.figure(figsize=[25,18])
    # plt.subplot(1,6,1)
    # plt.imshow(img,cmap='gray')
    #
    # plt.subplot(1,6,2)
    # plt.imshow(img_blur,cmap='gray')
    #
    # plt.subplot(1,6,3)
    # plt.imshow(img_sh,cmap='gray')
    #
    # plt.subplot(1,6,4)
    # plt.imshow(flipr,cmap='gray')
    #
    # plt.subplot(1,6,5)
    # plt.imshow(flip_blur,cmap='gray')
    #
    # plt.subplot(1,6,6)
    # plt.imshow(flip_sh,cmap='gray')
    # plt.show()

    # shapes.append(img.shape)
    # plt.imshow(img)
    # plt.show()




class Prepare_dataset():
    W = 353
    H = 50
    IMG_SIZE = 50
    REBUILD_DATA = True
    handwritten = "./data/handwritten-augmented"
    printed = "./data/printed"
    # TESTING = "PetImages/Testing"
    LABELS = {printed: 0 , handwritten: 1}
    training_data = []

    num_printed = 0
    num_handwritten = 0
    num_others = 0
    def make_training_data(self):
        for label in self.LABELS:
            # print(label)
            for f in tqdm.tqdm(os.listdir(label)):
                if "png" or "PNG" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
                        img = cv.resize(img, (self.W, self.H))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot
                        # print(np.eye(2)[self.LABELS[label]])
                        if label == self.printed:
                            self.num_printed += 1
                        elif label == self.handwritten:
                            self.num_handwritten += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))
        print(f"Number of printed images:\t{self.num_printed}\nNumber of Hadnwritten images:\t{self.num_handwritten}\n")

        np.random.shuffle(self.training_data)
        np.save("data/training_data.npy", self.training_data)


if REBUILD_DATA:
    dataset = Prepare_dataset()
    dataset.make_training_data()