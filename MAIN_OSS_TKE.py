#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:41:42 2024

@author: jitongd
"""

import numpy as np
import matplotlib.pyplot as plt
from read_file import get_xyzuvw_channelflow
from cheb_numeric import cheb
from derivative_calculation import fft_xy, get_3d_physical, get_kx_array_ky_array, getkm
from OSS_TKE import get_Orr_Sommerfeld_terms, get_Orr_Sommerfeld_D_terms, get_Squire_terms

Retau = 180
nx = 74
ny = 74
nz = 128
dkx = 1
dky = 2
nx_d = 32
ny_d = 32

D, z = cheb(nz)
D1 = D[1:-1, 1:-1]  
D2 = np.matmul(D, D)
D2 = D2[1:-1, 1:-1]
S = np.diag([0] + [1 / (1 - z[i] ** 2) for i in range(1, nz)] + [0])
D4 = (np.diag(1 - z**2) @ np.linalg.matrix_power(D, 4)
      - 8 * np.diag(z) @ np.linalg.matrix_power(D, 3)
      - 12 * np.matmul(D, D)) @ S
D4 = D4[1:nz, 1:nz]

Retau = 180
filename = f'full{Retau}_mean.npz'
data = np.load(filename, allow_pickle=True)
channelRe = data['channelRe'].item()
U      = channelRe['Up']
dUdz   = channelRe['Up_diff1'][1:-1]
d2Udz2 = channelRe['Up_diff2'][1:-1]

kx_array, ky_array = get_kx_array_ky_array(nx_d,ny_d,dkx,dky)
kx_m, ky_m         = getkm(kx_array, ky_array, nx, ny, dkx, dky)

kx_index = 1
ky_index = 1
kx = kx_m[ky_index,kx_index]
ky = ky_m[ky_index,kx_index]

L  = D2 - (kx**2 + ky**2) * np.eye(nz-1)
L2 = D4 + (kx**2 + ky**2)**2 * np.eye(nz-1) - 2 * (kx**2 + ky**2) * D2


filename = 'U100.nc'

x,y,z,u,v,w = get_xyzuvw_channelflow(filename)

u = u[1:-1,:,:]
v = v[1:-1,:,:]
w = w[1:-1,:,:]
u = u - U[1:-1, np.newaxis, np.newaxis]

u_F = fft_xy(u)
v_F = fft_xy(v)
w_F = fft_xy(w)

dudx,dudy,dudz = get_3d_physical(u, D1, nx_d, ny_d, nx, ny, dkx, dky)
dvdx,dvdy,dvdz = get_3d_physical(v, D1, nx_d, ny_d, nx, ny, dkx, dky)
dwdx,dwdy,dwdz = get_3d_physical(w, D1, nx_d, ny_d, nx, ny, dkx, dky)

fx = -u*dudx - v*dudy - w*dudz
fy = -u*dvdx - v*dvdy - w*dvdz
fz = -u*dwdx - v*dwdy - w*dwdz

fx_F = fft_xy(fx)
fy_F = fft_xy(fy)
fz_F = fft_xy(fz)

u_F  = u_F[:,ky_index,kx_index]
v_F  = v_F[:,ky_index,kx_index]
w_F  = w_F[:,ky_index,kx_index]
fx_F = fx_F[:,ky_index,kx_index]
fy_F = fy_F[:,ky_index,kx_index]
fz_F = fz_F[:,ky_index,kx_index]

OS_first_term,  OS_second_term,  OS_third_term,  OS_fourth_term,  OS_fifth_term,  OS_sixth_term  = get_Orr_Sommerfeld_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz)
OSD_first_term, OSD_second_term, OSD_third_term, OSD_fourth_term, OSD_fifth_term, OSD_sixth_term = get_Orr_Sommerfeld_D_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz)
S_first_term,   S_second_term,   S_third_term,   S_fourth_term,   S_fifth_term                   = get_Squire_terms(u_F,v_F,w_F,fx_F,fy_F,kx,ky,U,dUdz,L,D1,Retau,nz)

plt.plot(z[1:-1],np.real(OSD_third_term))