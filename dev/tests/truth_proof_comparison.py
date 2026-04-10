import numpy as np
import cv2
import os
import sys

# Implementation of MATLAB-style metrics (Exactly matching the formulas in metrics/ folder)
def matlab_entropy(F):
    F = np.floor(F).astype(np.int32)
    m, n = F.shape
    h = np.zeros(256)
    for i in range(m):
        for j in range(n):
            val = F[i, j]
            if 0 <= val < 256:
                h[val] += 1
    p = h / (m * n)
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))

def matlab_variance(F):
    # metricsVariance.m actually returns Standard Deviation!
    m, n = F.shape
    mu = np.mean(F)
    return np.sqrt(np.sum((F - mu)**2) / (m * n))

def matlab_qabf(A, B, F):
    # Parameters from metricsQabf.m
    Tg=0.9994; kg=-15; Dg=0.5; Ta=0.9879; ka=-22; Da=0.8
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    def get_grad(img):
        # imfilter with replicate is similar to BORDER_REPLICATE
        gx = cv2.filter2D(img.astype(np.float64), -1, h3, borderType=cv2.BORDER_CONSTANT)
        gy = cv2.filter2D(img.astype(np.float64), -1, h1, borderType=cv2.BORDER_CONSTANT)
        g = np.sqrt(gx**2 + gy**2)
        a = np.zeros_like(gx)
        mask = (gx != 0)
        a[mask] = np.arctan(gy[mask] / gx[mask])
        a[~mask] = np.pi / 2
        return g, a

    gA, aA = get_grad(A); gB, aB = get_grad(B); gF, aF = get_grad(F)
    
    def rel_strength(gS, gF):
        GAF = np.zeros_like(gS)
        m1 = gS > gF
        GAF[m1] = gF[m1] / gS[m1]
        m2 = gS == gF
        GAF[m2] = gF[m2] # Weird Matlab logic
        m3 = gS < gF
        GAF[m3] = gS[m3] / gF[m3]
        return GAF

    GAF = rel_strength(gA, gF); GBF = rel_strength(gB, gF)
    AAF = 1 - np.abs(aA - aF) / (np.pi/2); ABF = 1 - np.abs(aB - aF) / (np.pi/2)
    
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg))); QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QgBF = Tg / (1 + np.exp(kg * (GBF - Dg))); QaBF = Ta / (1 + np.exp(ka * (ABF - Da)))
    
    QAF = QgAF * QaAF; QBF = QgBF * QaBF
    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)
    return nume / deno if deno > 0 else 0

# Original Python implementations (before my fixes, based on files I saw)
def original_variance(F):
    return np.var(F.astype(np.float64))

def original_qg(A, B, F):
    # The one I saw before my big rewrite
    # It used np.arctan2 and different parameters
    kg=15; tg=0.5; ka=15; ta=0.75
    sx = cv2.Sobel(A.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(A.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gA = np.sqrt(sx**2 + sy**2); aA = np.arctan2(sx, sy)
    # ... (skipping full implementation for brevity, just to show major difference)
    return 0.0 # Placeholder for "different"

def compare():
    # Load same image
    base_path = r"Havard-Medical-Image-Fusion-Datasets-main\Havard-Medical-Image-Fusion-Datasets-main\SPECT-MRI"
    A = cv2.imread(os.path.join(base_path, "MRI", "25015.png"), 0)
    B = cv2.imread(os.path.join(base_path, "SPECT", "25015.png"), 0)
    F = cv2.imread(r"results\CDDFuse\25015.png", 0)
    
    if A is None: return
    
    print(f"{'Metric':15} | {'MATLAB Logic':15} | {'Fixed Python':15} | {'Match?'}")
    print("-" * 65)
    
    # Entropy
    m_ent = matlab_entropy(F)
    import metric as m
    p_ent = m.entropy(F)
    print(f"{'Entropy':15} | {m_ent:15.8f} | {p_ent:15.8f} | {abs(m_ent-p_ent)<1e-10}")
    
    # Variance
    m_var = matlab_variance(F)
    p_var = m.variance(F)
    print(f"{'Var (SD)':15} | {m_var:15.8f} | {p_var:15.8f} | {abs(m_var-p_var)<1e-10}")
    
    # QG
    m_qg = matlab_qabf(A, B, F)
    p_qg = m.qg_petrovic(A, B, F)
    print(f"{'Qabf (QG)':15} | {m_qg:15.8f} | {p_qg:15.8f} | {abs(m_qg-p_qg)<1e-10}")

if __name__ == "__main__":
    compare()
