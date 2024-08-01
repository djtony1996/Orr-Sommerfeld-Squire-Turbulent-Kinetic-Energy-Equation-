#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:04:46 2024

@author: jitongd
"""

import numpy as np

def get_Orr_Sommerfeld_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz):

    first_term = np.linalg.solve(L, (-1j * kx * np.diag(U[1:-1]) @ L) @ w_F * np.conj(w_F)) + \
                 np.linalg.solve(L, (1j * kx * np.diag(U[1:-1]) @ L) @ np.conj(w_F) * w_F)
    
    second_term = np.linalg.solve(L, (1j * kx * np.diag(d2Udz2)) @ w_F * np.conj(w_F)) + \
                  np.linalg.solve(L, (-1j * kx * np.diag(d2Udz2)) @ np.conj(w_F) * w_F)
    
    third_term = np.linalg.solve(L, (L2 / Retau) @ w_F * np.conj(w_F)) + \
                 np.linalg.solve(L, (L2 / Retau) @ np.conj(w_F) * w_F)
    
    fourth_term = np.linalg.solve(L, (-1j * kx * D1) @ fx_F * np.conj(w_F)) + \
                  np.linalg.solve(L, (1j * kx * D1) @ np.conj(fx_F) * w_F)
    
    fifth_term = np.linalg.solve(L, (-1j * ky * D1) @ fy_F * np.conj(w_F)) + \
                 np.linalg.solve(L, (1j * ky * D1) @ np.conj(fy_F) * w_F)
    
    sixth_term = np.linalg.solve(L, (-(kx**2 + ky**2) * np.eye(nz-1)) @ fz_F * np.conj(w_F)) + \
                 np.linalg.solve(L, (-(kx**2 + ky**2) * np.eye(nz-1)) @ np.conj(fz_F) * w_F)
    
    first_term  /= 2
    second_term /= 2
    third_term  /= 2
    fourth_term /= 2
    fifth_term  /= 2
    sixth_term  /= 2
    
    return first_term, second_term, third_term, fourth_term, fifth_term, sixth_term




def get_Orr_Sommerfeld_D_terms(w_F,fx_F,fy_F,fz_F,kx,ky,U,d2Udz2,L,L2,D1,Retau,nz):

    first_term = (D1 @ np.linalg.solve(L, (-1j * kx * np.diag(U[1:-1]) @ L) @ w_F * (D1 @ np.conj(w_F))) +
                  D1 @ np.linalg.solve(L, (1j * kx * np.diag(U[1:-1]) @ L) @ np.conj(w_F) * (D1 @ w_F)))
    
    second_term = (D1 @ np.linalg.solve(L, (1j * kx * np.diag(d2Udz2)) @ w_F * (D1 @ np.conj(w_F))) +
                   D1 @ np.linalg.solve(L, (-1j * kx * np.diag(d2Udz2)) @ np.conj(w_F) * (D1 @ w_F)))
    
    third_term = (D1 @ np.linalg.solve(L, (L2 / Retau) @ w_F * (D1 @ np.conj(w_F))) +
                  D1 @ np.linalg.solve(L, (L2 / Retau) @ np.conj(w_F) * (D1 @ w_F)))
    
    fourth_term = (D1 @ np.linalg.solve(L, (-1j * kx * D1) @ fx_F * (D1 @ np.conj(w_F))) +
                   D1 @ np.linalg.solve(L, (1j * kx * D1) @ np.conj(fx_F) * (D1 @ w_F)))
    
    fifth_term = (D1 @ np.linalg.solve(L, (-1j * ky * D1) @ fy_F * (D1 @ np.conj(w_F))) +
                  D1 @ np.linalg.solve(L, (1j * ky * D1) @ np.conj(fy_F) * (D1 @ w_F)))
    
    sixth_term = (D1 @ np.linalg.solve(L, (-(kx**2 + ky**2) * np.eye(nz-1)) @ fz_F * (D1 @ np.conj(w_F))) +
                  D1 @ np.linalg.solve(L, (-(kx**2 + ky**2) * np.eye(nz-1)) @ np.conj(fz_F) * (D1 @ w_F)))
    
    kx_ky_sq = kx**2 + ky**2
    first_term  /= 2 * kx_ky_sq
    second_term /= 2 * kx_ky_sq
    third_term  /= 2 * kx_ky_sq
    fourth_term /= 2 * kx_ky_sq
    fifth_term  /= 2 * kx_ky_sq
    sixth_term  /= 2 * kx_ky_sq
    
    return first_term, second_term, third_term, fourth_term, fifth_term, sixth_term



def get_Squire_terms(u_F,v_F,w_F,fx_F,fy_F,kx,ky,U,dUdz,L,D1,Retau,nz):
    eta_zF = 1j * ky * u_F - 1j * kx * v_F
    
    first_term  = (-1j * ky * np.diag(dUdz) @ w_F * np.conj(eta_zF) +
                   1j * ky * np.diag(dUdz) @ np.conj(w_F) * eta_zF)
    
    second_term = (-1j * kx * np.diag(U[1:-1]) @ eta_zF * np.conj(eta_zF) +
                   1j * kx * np.diag(U[1:-1]) @ np.conj(eta_zF) * eta_zF)
    
    third_term  = (L @ eta_zF / Retau * np.conj(eta_zF) +
                   L @ np.conj(eta_zF) / Retau * eta_zF)
    
    fourth_term = (1j * ky * fx_F * np.conj(eta_zF) - 
                   1j * ky * np.conj(fx_F) * eta_zF)
    
    fifth_term  = (-1j * kx * fy_F * np.conj(eta_zF) +
                   1j * kx * np.conj(fy_F) * eta_zF)
    
    kx_ky_sq = kx**2 + ky**2
    first_term  /= 2 * kx_ky_sq
    second_term /= 2 * kx_ky_sq
    third_term  /= 2 * kx_ky_sq
    fourth_term /= 2 * kx_ky_sq
    fifth_term  /= 2 * kx_ky_sq
    
    return first_term, second_term, third_term, fourth_term, fifth_term

