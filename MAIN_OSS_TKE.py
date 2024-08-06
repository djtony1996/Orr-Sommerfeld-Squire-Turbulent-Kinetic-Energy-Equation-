#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:41:42 2024

@author: jitongd
"""

import sys
import numpy as np
from read_file import get_xyzuvw_channelflow
from cheb_numeric import cheb
from derivative_calculation import fft_xy, get_3d_physical, get_kx_array_ky_array, getkm
from OSS_TKE import get_Orr_Sommerfeld_terms, get_Orr_Sommerfeld_D_terms, get_Squire_terms

kx_index = int(sys.argv[1])
ky_index = int(sys.argv[2])
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

kx = kx_m[ky_index,kx_index]
ky = ky_m[ky_index,kx_index]

L  = D2 - (kx**2 + ky**2) * np.eye(nz-1)
L2 = D4 + (kx**2 + ky**2)**2 * np.eye(nz-1) - 2 * (kx**2 + ky**2) * D2

OS_first_avg   = np.zeros([nz-1])
OS_second_avg  = np.zeros([nz-1])
OS_third_avg   = np.zeros([nz-1])
OS_fourth_avg  = np.zeros([nz-1])
OS_fifth_avg   = np.zeros([nz-1])
OS_sixth_avg   = np.zeros([nz-1])
OSD_first_avg  = np.zeros([nz-1])
OSD_second_avg = np.zeros([nz-1])
OSD_third_avg  = np.zeros([nz-1])
OSD_fourth_avg = np.zeros([nz-1])
OSD_fifth_avg  = np.zeros([nz-1])
OSD_sixth_avg  = np.zeros([nz-1])
S_first_avg    = np.zeros([nz-1])
S_second_avg   = np.zeros([nz-1])
S_third_avg    = np.zeros([nz-1])
S_fourth_avg   = np.zeros([nz-1])
S_fifth_avg    = np.zeros([nz-1])

read_array = np.arange(100,1101,1)

for k_array in range(len(read_array)):
    filename = f"data180/uTot/U{read_array[k_array]}.nc"
    
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
    
    OS_first,  OS_second,  OS_third,  OS_fourth,  OS_fifth,  OS_sixth  = get_Orr_Sommerfeld_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz)
    OSD_first, OSD_second, OSD_third, OSD_fourth, OSD_fifth, OSD_sixth = get_Orr_Sommerfeld_D_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz)
    S_first,   S_second,   S_third,   S_fourth,   S_fifth              = get_Squire_terms(u_F,v_F,w_F,fx_F,fy_F,kx,ky,U,dUdz,L,D1,Retau,nz)

    OS_first_avg  = OS_first_avg*k_array/(k_array+1)  + OS_first/(k_array+1)
    OS_second_avg = OS_second_avg*k_array/(k_array+1) + OS_second/(k_array+1)
    OS_third_avg  = OS_third_avg*k_array/(k_array+1)  + OS_third/(k_array+1)
    OS_fourth_avg = OS_fourth_avg*k_array/(k_array+1) + OS_fourth/(k_array+1)
    OS_fifth_avg  = OS_fifth_avg*k_array/(k_array+1)  + OS_fifth/(k_array+1)
    OS_sixth_avg  = OS_sixth_avg*k_array/(k_array+1)  + OS_sixth/(k_array+1)
    
    OSD_first_avg  = OSD_first_avg*k_array/(k_array+1)  + OSD_first/(k_array+1)
    OSD_second_avg = OSD_second_avg*k_array/(k_array+1) + OSD_second/(k_array+1)
    OSD_third_avg  = OSD_third_avg*k_array/(k_array+1)  + OSD_third/(k_array+1)
    OSD_fourth_avg = OSD_fourth_avg*k_array/(k_array+1) + OSD_fourth/(k_array+1)
    OSD_fifth_avg  = OSD_fifth_avg*k_array/(k_array+1)  + OSD_fifth/(k_array+1)
    OSD_sixth_avg  = OSD_sixth_avg*k_array/(k_array+1)  + OSD_sixth/(k_array+1)

    S_first_avg  = S_first_avg*k_array/(k_array+1)  + S_first/(k_array+1)
    S_second_avg = S_second_avg*k_array/(k_array+1) + S_second/(k_array+1)
    S_third_avg  = S_third_avg*k_array/(k_array+1)  + S_third/(k_array+1)
    S_fourth_avg = S_fourth_avg*k_array/(k_array+1) + S_fourth/(k_array+1)
    S_fifth_avg  = S_fifth_avg*k_array/(k_array+1)  + S_fifth/(k_array+1)


if kx != 0 and ky != 0:
    kx = -kx
    kx_index = -kx_index
    
    for k_array in range(len(read_array)):
        filename = f"data180/uTot/U{read_array[k_array]}.nc"
        
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
        
        OS_first,  OS_second,  OS_third,  OS_fourth,  OS_fifth,  OS_sixth  = get_Orr_Sommerfeld_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz)
        OSD_first, OSD_second, OSD_third, OSD_fourth, OSD_fifth, OSD_sixth = get_Orr_Sommerfeld_D_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz)
        S_first,   S_second,   S_third,   S_fourth,   S_fifth              = get_Squire_terms(u_F,v_F,w_F,fx_F,fy_F,kx,ky,U,dUdz,L,D1,Retau,nz)

        OS_first_avg  = OS_first_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OS_first/(k_array+1+len(read_array))
        OS_second_avg = OS_second_avg*(k_array+len(read_array))/(k_array+1+len(read_array)) + OS_second/(k_array+1+len(read_array))
        OS_third_avg  = OS_third_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OS_third/(k_array+1+len(read_array))
        OS_fourth_avg = OS_fourth_avg*(k_array+len(read_array))/(k_array+1+len(read_array)) + OS_fourth/(k_array+1+len(read_array))
        OS_fifth_avg  = OS_fifth_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OS_fifth/(k_array+1+len(read_array))
        OS_sixth_avg  = OS_sixth_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OS_sixth/(k_array+1+len(read_array))
        
        OSD_first_avg  = OSD_first_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OSD_first/(k_array+1+len(read_array))
        OSD_second_avg = OSD_second_avg*(k_array+len(read_array))/(k_array+1+len(read_array)) + OSD_second/(k_array+1+len(read_array))
        OSD_third_avg  = OSD_third_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OSD_third/(k_array+1+len(read_array))
        OSD_fourth_avg = OSD_fourth_avg*(k_array+len(read_array))/(k_array+1+len(read_array)) + OSD_fourth/(k_array+1+len(read_array))
        OSD_fifth_avg  = OSD_fifth_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OSD_fifth/(k_array+1+len(read_array))
        OSD_sixth_avg  = OSD_sixth_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + OSD_sixth/(k_array+1+len(read_array))

        S_first_avg  = S_first_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + S_first/(k_array+1+len(read_array))
        S_second_avg = S_second_avg*(k_array+len(read_array))/(k_array+1+len(read_array)) + S_second/(k_array+1+len(read_array))
        S_third_avg  = S_third_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + S_third/(k_array+1+len(read_array))
        S_fourth_avg = S_fourth_avg*(k_array+len(read_array))/(k_array+1+len(read_array)) + S_fourth/(k_array+1+len(read_array))
        S_fifth_avg  = S_fifth_avg*(k_array+len(read_array))/(k_array+1+len(read_array))  + S_fifth/(k_array+1+len(read_array))


kx_int = int(kx)
ky_int = int(ky)
savename = f'OSS_TKE_onewavenumber{kx_int}_{ky_int}.npz'
np.savez(savename,
         OS_first_avg=OS_first_avg, OS_second_avg=OS_second_avg, OS_third_avg=OS_third_avg, OS_fourth_avg=OS_fourth_avg, OS_fifth_avg=OS_fifth_avg, OS_sixth_avg=OS_sixth_avg, 
         OSD_first_avg=OSD_first_avg, OSD_second_avg=OSD_second_avg, OSD_third_avg=OSD_third_avg, OSD_fourth_avg=OSD_fourth_avg, OSD_fifth_avg=OSD_fifth_avg, OSD_sixth_avg=OSD_sixth_avg, 
         S_first_avg=S_first_avg, S_second_avg=S_second_avg, S_third_avg=S_third_avg, S_fourth_avg=S_fourth_avg, S_fifth_avg=S_fifth_avg)
