from umucv.htrans import htrans, desp, scale
import numpy as np

T = np.array([[9.34339394e-01, -9.02304222e-02, 5.32311208e+01],
              [1.20861120e-01, 7.48669047e-01, -5.08686770e+01],
              [2.76452768e-04, -7.54448078e-05, 1.00000000e+00]])

IH = np.linalg.inv(T)
print("T=",T)
print("IH=",IH)

q = htrans(T, [100,100])
print("Q=",q)

p = htrans(IH, q)
print("p=",p)