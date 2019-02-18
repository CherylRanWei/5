from __future__ import division
import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from skimage import img_as_ubyte

print '************ Problem 1 **************'
img = io.imread('5_1.bmp')
im = img.astype(float) / 255.0 # apply fft
fft_f = np.fft.fft2(im)
f = np.fft.fftshift(fft_f)
magnitudeF = np.log(np.abs(f))
angleF = np.angle(f)

magnitudeF[110:150, 110:150] = 0
magnitudeF[110:150, 360:400] = 0
magnitudeF[360:400, 110:150] = 0
magnitudeF[360:400, 360:400] = 0

mag = np.exp(magnitudeF)
cosine = np.zeros(img.shape)
sine = np.zeros(img.shape)

for i in range(angleF.shape[0]):
    for j in range(angleF.shape[1]):
        cosine[i,j] = math.cos(angleF[i,j])
        sine[i,j] = math.sin(angleF[i,j])


inverse_f =  (mag) * (cosine + sine*1j)
inferse_fft_f = np.fft.ifftshift(inverse_f)
inverse_im = np.fft.ifft2(inferse_fft_f)
inverse_im = inverse_im.astype(float)


plt.figure()
plt.imshow(img, cmap = 'gray')
plt.title('Input Image')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(magnitudeF, cmap = 'gray')
plt.title('Magnitude Spectrum')

plt.figure()
plt.imshow(angleF, cmap = 'gray')
plt.title('Phase Spectrum')
plt.xticks([])
plt.yticks([])

plt.figure()
plt.imshow(inverse_im, cmap = 'gray')
plt.title('Combined Image')
plt.xticks([])
plt.yticks([])

plt.show()



