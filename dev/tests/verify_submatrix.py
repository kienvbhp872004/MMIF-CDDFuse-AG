import numpy as np
import metric as m
import sys
import os

def test_new_metrics():
    # Create small 5x5 sub-matrices for testing
    np.random.seed(42)
    A = np.random.randint(0, 256, (10, 10)).astype(np.float64)
    B = np.random.randint(0, 256, (10, 10)).astype(np.float64)
    F = (A + B) / 2.0  # Simple average fusion
    
    print("Testing New Metrics with 10x10 Sub-matrices:")
    print("-" * 50)
    
    # 1. Test NCIE
    try:
        val_ncie = m.ncie(A, B, F)
        print(f"NCIE (Python Generated): {val_ncie:.8f}")
    except Exception as e:
        print(f"NCIE Error: {str(e)}")
        
    # 2. Test NABF
    try:
        val_nabf = m.nabf(A, B, F)
        print(f"NABF (Python Polished) : {val_nabf:.8f}")
    except Exception as e:
        print(f"NABF Error: {str(e)}")

    # 3. Double check with existing QNCIE to see if they match
    try:
        val_qncie = m.ncc_entropy(A, B, F) # This is where QNCIE was mapped in __init__.py
        print(f"Existing QNCIE         : {val_qncie:.8f}")
    except:
        pass

if __name__ == "__main__":
    test_new_metrics()
