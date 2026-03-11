import numpy as np
import cv2


def gaussian_blur(img,sigma=2):

    return cv2.GaussianBlur(img,(0,0),sigma)


def peli_contrast(img):

    Ik = gaussian_blur(img,2)
    Ik1 = gaussian_blur(img,4)

    C = (Ik-img)/(Ik1-img+1e-8)-1
    return C


def masked_contrast(C,t=1,h=1,p=1,q=1,Z=1e-6):

    return (t*(C**p))/(h*(C**q)+Z)


def QCB(A,B,F):

    CA = peli_contrast(A)
    CB = peli_contrast(B)
    CF = peli_contrast(F)

    CA = masked_contrast(CA)
    CB = masked_contrast(CB)
    CF = masked_contrast(CF)

    A_map = (CA**2)/(CA**2 + CB**2 + 1e-8)
    B_map = 1-A_map

    QAF = np.minimum(CA,CF)/np.maximum(CA,CF)
    QBF = np.minimum(CB,CF)/np.maximum(CB,CF)

    QGQM = A_map*QAF + B_map*QBF

    return np.mean(QGQM)