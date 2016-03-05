#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale, iradon
from skimage.draw import circle
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter

parser = argparse.ArgumentParser(description="Simulation of Green's one step late algorithm with iteratively reweighted shrinkage")
parser.add_argument('--count', '-c', default=2e6, type=float,
                    help='slice total count. Poisson noise equivalent to COUNT is added to sinogram. If COUNT is zero, no noise is added to sinogram (true).')
parser.add_argument('--niter', '-i', default=10, type=int,
                    help='number of iteration')
parser.add_argument('--nsub', '-s', default=10, type=int,
                    help='number of subset')
parser.add_argument('--filter', '-f', default=1.5, type=float,
                    help='smoothing filer FWHM [pix]')
parser.add_argument('--beta', '-b', default=0.02, type=float,
                    help='prior strength')
parser.add_argument('--median', action='store_true',
                    help='median root prior')
args = parser.parse_args()

count   = args.count
niter   = args.niter
nsub    = args.nsub
sfwhm   = args.filter
beta    = args.beta
median  = args.median

# shepp-logan phantom
image   = imread(data_dir + "/phantom.png", as_grey=True)
image   = rescale(image, 0.4)
shape   = image.shape

# sinogram
theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram    = radon(image, theta=theta, circle=True)

# add noise
if count > 0:
    val = sinogram.sum()
    sinogram    = np.random.poisson(sinogram / val * count).astype(np.float)
    sinogram    *= val / count

# normalization matrix
nview   = len(theta)
norm    = np.ones(shape)
wgts    = []
for sub in xrange(nsub):
    views   = range(sub, nview, nsub)
    wgt = iradon(norm[:, views], theta=theta[views], filter=None, circle=True)
    wgts.append(wgt)

# comparison OSEM, IRS
recons  = []
for b, rw, m in [(0, False, False), (beta, True, False), (beta, True, median)]:
    print   'beta', b, 'rw', rw
    
    # initial
    recon   = np.zeros(shape)
    rr, cc  = circle(shape[0] / 2, shape[1] / 2, shape[0] / 2 - 1)
    recon[rr, cc]   = 1

    if rw:
        weight  = np.ones(shape)

    # iteration
    for iter in xrange(niter):
        print   'iter', iter
        order   = np.random.permutation(range(nsub))
        for sub in order:
            views   = range(sub, nview, nsub)
            fp  = radon(recon, theta=theta[views], circle=True)
            ratio   = sinogram[:, views] / (fp + 1e-6)
            bp  = iradon(ratio, theta=theta[views], filter=None, circle=True)
        
            if b > 0:
                if m:
                    regular = (recon - median_filter(recon, 3)) * b
                else:
                    regular = (recon - uniform_filter(recon)) * b
                if rw:
                    regular *= 0.5 / (weight + 1e-6)
                recon   *= bp / (wgts[sub] + regular + 1e-6)
                if rw:
                    if m:
                        weight  = np.abs(recon - median_filter(recon, 3))
                    else:
                        weight  = np.abs(recon - uniform_filter(recon))
            else:
                recon   *= bp / (wgts[sub] + 1e-6)

    recons.append(recon.copy())

fbp = iradon(sinogram, theta=theta, circle=True)
if sfwhm > 0:
    fbp = gaussian_filter(fbp, sfwhm / 2.355)
    recons[0]    = gaussian_filter(recons[0], sfwhm / 2.355)

# display
plt.figure(figsize = (10, 5))
plt.gray()
plt.subplot(141)
plt.title('FBP\nfilter = {} [pix]\nRMSE = {:.3f}'.format(sfwhm, np.sqrt((fbp - image) ** 2).mean()))
plt.imshow(fbp, vmin = 0, vmax = 1)
plt.axis('off')
plt.subplot(142)
plt.title('OSEM\nfilter = {} [pix]\nRMSE = {:.3f}'.format(sfwhm, np.sqrt((recons[0] - image) ** 2).mean()))
plt.imshow(recons[0], vmax = 1)
plt.axis('off')
plt.subplot(143)
plt.title('OSL-IRS\nbeta = {}\nRMSE = {:.3f}'.format(beta, np.sqrt((recons[1] - image) ** 2).mean()))
plt.imshow(recons[1], vmax = 1)
plt.axis('off')
plt.subplot(144)
plt.title('OSL-IRS-Median\nbeta = {}\nRMSE = {:.3f}'.format(beta, np.sqrt((recons[2] - image) ** 2).mean()))
plt.imshow(recons[2], vmax = 1)
plt.axis('off')

plt.subplots_adjust(0, 0, 1, 0.9, 0, 0.2)
plt.savefig('osl.png')

plt.figure()
plt.suptitle('Profile of image')
plt.plot(image[shape[0] / 2 - 3:shape[0] / 2 + 3].mean(axis = 0), label='Phantom')
plt.plot(fbp[shape[0] / 2 - 3:shape[0] / 2 + 3].mean(axis = 0), label='FBP')
plt.plot(recons[0][shape[0] / 2 - 3:shape[0] / 2 + 3].mean(axis = 0), label='OSEM')
plt.plot(recons[1][shape[0] / 2 - 3:shape[0] / 2 + 3].mean(axis = 0), label='OSL-IRS')
plt.plot(recons[2][shape[0] / 2 - 3:shape[0] / 2 + 3].mean(axis = 0), label='OSL-IRS-Median')
plt.legend(loc=3, prop={'size':10})
plt.xlabel('X [pix]')
plt.ylabel('[a.u.]')
plt.savefig('profile.png')

plt.show()
