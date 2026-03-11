import numpy as np
import cv2
from scipy.signal import convolve2d


def sobel_edges(img):

    gx = cv2.Sobel(img, cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F,0,1,ksize=3)

    return np.sqrt(gx**2 + gy**2)


def csf_filter(img):

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    return convolve2d(img, kernel, mode="same")


def QCV(A,B,F,window=8):

    GA = sobel_edges(A)
    GB = sobel_edges(B)

    h,w = A.shape

    num = 0
    den = 0

    for i in range(0,h-window,window):
        for j in range(0,w-window,window):

            Aw = A[i:i+window,j:j+window]
            Bw = B[i:i+window,j:j+window]
            Fw = F[i:i+window,j:j+window]

            GAw = GA[i:i+window,j:j+window]
            GBw = GB[i:i+window,j:j+window]

            salA = np.sum(GAw**2)
            salB = np.sum(GBw**2)

            diffA = csf_filter(Aw-Fw)
            diffB = csf_filter(Bw-Fw)

            DA = np.mean(diffA**2)
            DB = np.mean(diffB**2)

            num += salA*DA + salB*DB
            den += salA + salB

    return num/den