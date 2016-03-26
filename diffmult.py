# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:33:01 2016

@author: laura
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import linalg
from numpy.linalg import solve, eig
from time import localtime
from scipy.io import savemat
__docformat__ = 'restructuredtext fr'
# Tout en um


def drawPositions(Nat = 200,Rad = 25.,dz = 0.2,core = 0.05,avoid=True,MethodZ = 1):
    """
    Tirage des positions des atomes
    
    Parameters
    ----------
    
    Nat  : nombre d'atomes (defaut 200)
    Rad  : rayon du disque dans lequel les positions sont tirees (defaut 25um)
    dz   : epaisseur du gaz (tirage uniforme) (defaut 0.2um)
    core : taille de la "sphere dure" de chaque atome (pas encore implemente)
    
    Returns
    -------
    
    R : matrice Nat x 3 donnant x, y, z en um
    
    """
    
    R = np.zeros((Nat,3))
    
    if not avoid:
        for i in range(Nat):
            r = np.sqrt(np.random.rand())*Rad
            theta = 2*np.pi*np.random.rand()
            
            x0 = r*np.cos(theta)
            y0 = r*np.sin(theta)
            if MethodZ == 0:
                z0 = dz*(np.random.rand()-0.5)
            elif MethodZ==1:
                z0 = dz*np.random.randn()
            
            R[i,0] = x0
            R[i,1] = y0
            R[i,2] = z0
    else:
        for i in range(Nat):
            isGood = False
            count = 0;
            while (not isGood) and count < 500:
                r = np.sqrt(np.random.rand())*Rad
                theta = 2*np.pi*np.random.rand()
                
                x0 = r*np.cos(theta)
                y0 = r*np.sin(theta)
                if MethodZ == 0:
                    z0 = dz*(np.random.rand()-0.5)
                elif MethodZ==1:
                    z0 = dz*np.random.randn()
                    
                oneProblem = False
                for i2 in range(i):
                    x1 = R[i2,0]
                    y1 = R[i2,1]
                    z1 = R[i2,2]
                    d = np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
                    if d < core:
                        oneProblem = True
                        break
                if not oneProblem:
                    isGood = True
                count = count + 1
            if count == 500:
                print 'Maximum counts reached'
                break
            
            R[i,0] = x0
            R[i,1] = y0
            R[i,2] = z0
        
    
    return R
    
# Delta en unites de Gamma/2
def Heff(R,Delta=0,k0=2*np.pi/0.78):
    """
    Calcul de la matrice à inverser h nu_laser - Heff
    
    Parameters
    ----------
    
    R     : matrice Nat x 3 donnant la position des atomes en um
    Delta : désaccord de la sonde en unités de Gamma/2
    k0    : module du vecteur d'onde associé à la lumière
    
    Returns
    -------
    
    H : matrice 3Nat x 3Nat donnant h nu_laser - Heff en unité de hbar Gamma/2
    
    """
    Nat = R.shape[0]
    
#    Initialisation de la matrice et remplissage de la diagonale
    H = np.zeros((3*Nat,3*Nat),dtype=np.complex64)
    forDiag = (Delta+1j)*np.eye(3*Nat,dtype=np.complex64)
    H = H + forDiag
    
#    Récupération des position et calcul des Ri - Rj
    x = R[:,0]
    y = R[:,1]
    z = R[:,2]
    
    xh = x[:, np.newaxis]
    yh = y[:, np.newaxis]
    zh = z[:, np.newaxis]
    
    xv = xh.T
    yv = yh.T
    zv = zh.T
    
    dx = (xv-xh)
    dy = (yv-yh)
    dz = (zv-zh)  
    
#    Calcul du tenseur R x R
    rr = np.kron(dx*dx,[[1,0,0],[0,0,0],[0,0,0]]) + np.kron(dy*dy,[[0,0,0],[0,1,0],[0,0,0]]) + np.kron(dz*dz,[[0,0,0],[0,0,0],[0,0,1]]) + np.kron(dx*dy,[[0,1,0],[1,0,0],[0,0,0]]) + np.kron(dx*dz,[[0,0,1],[0,0,0],[1,0,0]]) + np.kron(dz*dy,[[0,0,0],[0,0,1],[0,1,0]])
    
#    Calcul de la distance entre paires d'atomes
    dx2 = dx**2
    dy2 = dy**2
    dz2 = dz**2
    
    dist = np.sqrt(dx2 + dy2 + dz2);
    dist = dist.astype(dtype=np.complex64)
    distInv = 1./(dist + np.eye(Nat)) - np.eye(Nat)
    DistInv = np.kron(distInv,[[1,1,1],[1,1,1],[1,1,1]])
    
    RR = rr*(DistInv**2)
    
#    Calcul des g1 et g2 et complétion du Hamiltonien
    xx = dist * k0
    xxm3 = (xx + np.eye(Nat))**(-3) - np.eye(Nat)
    
    g1 = xxm3 * (-xx**2 - 1j*xx + 1) * np.exp(1j*xx)
    g1 = np.kron(g1,np.eye(3))
    H = H -1.5 * g1
    g2 = xxm3 * (xx**2 + 3*1j*xx - 3) * np.exp(1j*xx)
    g2 = np.kron(g2,np.ones((3,3))) * RR
    H = H -1.5 * g2

    return H
    
def test2atomes():
    """
    test des valeurs propres pour 2 atomes
    1000 points pris pour la distance entre 0.05 et 5 fois lambda
    
    
    Returns
    -------
    
    H : matrice 3Nat x 3Nat donnant la dernière valeur de h nu_laser - Heff en unité de hbar Gamma/2
    
    """
    ul = 0.78/(2*np.pi)
    npoints = 1000
    r = 0.78*np.linspace(0.05,5,npoints)/ul
    v1 = np.zeros((npoints,),dtype=np.complex64)
    v2 = np.zeros((npoints,),dtype=np.complex64)
    for i in range(npoints):
        R = np.asarray([[0,0,0],[0,0,r[i]]])
        H = Heff(R,k0 = 2*np.pi/0.78 * ul)
        w, v = eig(H)
        v1[i] = w[0]
        v2[i] = w[1]
        
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(r,np.imag(v1),'.')
    plt.plot(r,np.imag(v2),'.')
    plt.grid(b=True)
    plt.subplot(2,1,2)
    plt.plot(r,np.real(v1),'.')
    plt.plot(r,np.real(v2),'.')
    plt.ylim(ymin = -2, ymax = 2)
    plt.grid(b=True)
        
    return H
    
def excVector(R,pol = np.array([1,0,0],dtype=np.complex64),  k = np.array([0,0,1],dtype=np.complex64),k0=2*np.pi/0.78):
    """
    Calcul du vecteur d'excitation des atomes poyr inversion du système pour un éclairement homogène
    
    Parameters
    ----------
    
    R     : matrice Nat x 3 donnant la position des atomes en um
    pol   : polarisation de la lumière
    k     : direction de la propagation de la lumière (renormalisé dès le début du calcul)
    k0    : module du vecteur d'onde associé à la lumière
    
    Returns
    -------
    
    a : vecteur 1 x 3Nat polarisation * exp(i k . R)
    
    """
    Nat = R.shape[0]
    k = k0*k/np.sqrt(k.dot(k))
    K = np.tile(k,(Nat,1))
    KR = np.sum(K*R,1)
    expKR = np.exp(1j*KR)
    a = np.kron(expKR,pol)
    return a
    
def excVectorLocalized(R,Rmin = 0.5,Rmax = 0.6,pol = np.array([1,0,0],dtype=np.complex64),                                                        k = np.array([0,0,1],dtype=np.complex64),k0=2*np.pi/0.78):
    """
    Calcul du vecteur d'excitation des atomes poyr inversion du système pour un éclairement localisé à un disque 
    
    Parameters
    ----------
    
    R     : matrice Nat x 3 donnant la position des atomes en um
    Rmax  : rayon dans lequel les atomes sont excités
    pol   : polarisation de la lumière
    k     : direction de la propagation de la lumière (renormalisé dès le début du calcul)
    k0    : module du vecteur d'onde associé à la lumière
    
    Returns
    -------
    
    a : vecteur 1 x 3Nat polarisation * exp(i k . R)
    
    """
    distToCenter = np.sqrt(R[:,0]*R[:,0] + R[:,1]*R[:,1])
    goodAtoms = (distToCenter>Rmin)*(distToCenter<Rmax)
#    goodAtoms = distToCenter<Rmax
    
    Nat = R.shape[0]
    k = k0*k/np.sqrt(k.dot(k))
    K = np.tile(k,(Nat,1))
    KR = np.sum(K*R,1)
    expKR = np.exp(1j*KR)*(goodAtoms)
    a = np.kron(expKR,pol)
    return a
    
def recapVectorLocalized(R,Rmin = 0.5,Rmax = 30,pol = np.array([1,0,0],dtype=np.complex64),                                                        k = np.array([0,0,1],dtype=np.complex64),k0=2*np.pi/0.78):
    """
    Calcul du vecteur d'excitation des atomes poyr inversion du système pour un éclairement localisé à un disque 
    
    Parameters
    ----------
    
    R     : matrice Nat x 3 donnant la position des atomes en um
    Rmax  : rayon dans lequel les atomes sont excités
    pol   : polarisation de la lumière
    k     : direction de la propagation de la lumière (renormalisé dès le début du calcul)
    k0    : module du vecteur d'onde associé à la lumière
    
    Returns
    -------
    
    a : vecteur 1 x 3Nat polarisation * exp(i k . R)
    
    """
    distToCenter = np.sqrt(R[:,0]*R[:,0] + R[:,1]*R[:,1])
    goodAtoms = (distToCenter>Rmin)*(distToCenter<Rmax)
    
    Nat = R.shape[0]
    k = k0*k/np.sqrt(k.dot(k))
    K = np.tile(k,(Nat,1))
    KR = np.sum(K*R,1)
    expKR = np.exp(1j*KR)*(goodAtoms)
    a = np.kron(expKR,pol)
    return a
    
def transmisionOld(Nat=1000,Nrepeat=10,Rad=1.75,dz = 0.2,core = 0.5,Delta = 0.,Rmin=None,Rmax=None):
    """
    Calcul de l'élément de matrice <ki|T(Ei)|ki> pour un éclairement uniforme
    
    Parameters
    ----------
    
    Nat     : nombre d'atomes pour l'expérience
    Nrepeat : nombre de répétition du calcul de la transmission
    Rad     : rayon du disque dans lequel les positions sont tirees (defaut 1.75um)
    dz   : epaisseur du gaz (tirage uniforme) (defaut 0.2um)
    core : taille de la "sphere dure" de chaque atome (pas encore implemente)
    Delta : désaccord de la sonde en unités de Gamma/2
    
    Returns
    -------
    
    mr : moyenne de la partie réelle de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    mi : moyenne de la partie imaginaire de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    sr : écart-type de la partie réelle de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    si : écart-type de la partie imaginaire de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    
    """
    Tim = np.zeros((Nrepeat,),dtype=np.complex64)
    for i in range(Nrepeat):
        print str(i+1) + '/' + str(Nrepeat)
        R = drawPositions(Nat,Rad,dz,core)
        if type(Rmin) == type(None):
            a = excVector(R)
            aa = a.conj()
        else:
            if type(Rmax) == type(None):
                Rmax = Rad
            a = excVectorLocalized(R,Rmin)  
            aa = recapVectorLocalized(R,Rmin,Rmax)  
            aa = aa.conj()  
        H = Heff(R,Delta)
        Tim[i] = aa.dot(solve(H,a))
#        Tim[i] = a.dot(solve(H,a))
        
    mr = np.mean(np.real(Tim))
    mi = np.mean(np.imag(Tim))
    sr = np.std(np.real(Tim))
    si = np.std(np.imag(Tim))
        
    return mr,mi,sr,si
    
def transmision(Nat=1000,Nrepeat=10,Rad=1.75,dz = 0.2,core = 0.01,Delta = 0.,avoid=True,MethodZ = 1):
    """
    Calcul de l'élément de matrice <ki|T(Ei)|ki> pour un éclairement uniforme
    
    Parameters
    ----------
    
    Nat     : nombre d'atomes pour l'expérience
    Nrepeat : nombre de répétition du calcul de la transmission
    Rad     : rayon du disque dans lequel les positions sont tirees (defaut 1.75um)
    dz   : epaisseur du gaz (tirage uniforme) (defaut 0.2um)
    core : taille de la "sphere dure" de chaque atome (pas encore implemente)
    Delta : désaccord de la sonde en unités de Gamma/2
    
    Returns
    -------
    
    mr : moyenne de la partie réelle de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    mi : moyenne de la partie imaginaire de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    sr : écart-type de la partie réelle de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    si : écart-type de la partie imaginaire de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    
    """
    OD0 = Nat/(np.pi*Rad**2) * 3*0.78**2/(2*np.pi)
    Tim = np.zeros((Nrepeat,3),dtype=np.complex64)
    ODs = np.zeros((Nrepeat,),dtype=np.complex64)
    phis = np.zeros((Nrepeat,),dtype=np.complex64)
    for i in range(Nrepeat):
        print str(i+1) + '/' + str(Nrepeat)
        R = drawPositions(Nat,Rad,dz,core)
        ax = excVector(R,pol = np.array([1. + 0j,0,0]))
        ay = excVector(R,pol = np.array([0,1. + 0j,0]))
        az = excVector(R,pol = np.array([0,0,1. + 0j]))
        H = Heff(R,Delta)
        bx = solve(H,ax)
        Tim[i,0] = 1-0.5*1j*OD0*ax.conj().dot(bx)/Nat
        Tim[i,1] = -0.5*1j*OD0*ay.conj().dot(bx)/Nat
        Tim[i,2] = -0.5*1j*OD0*az.conj().dot(bx)/Nat
        ODs[i] = -2*np.log(np.abs(1-0.5*1j*OD0*ax.conj().dot(bx)/Nat))
        phis[i] = np.angle(Tim[i,0])
        
    trans = np.mean(Tim,0)
    transStd = np.std(Tim,0)
    OD = np.mean(ODs)
    ODstd = np.std(ODs)
    phi = np.mean(phis)
    phistd = np.std(phis)
        
    return trans,transStd,OD,ODstd,phi,phistd
    
def manyExpVaryN(Nat=np.array([10,50,75,100,250,500,750,1000]),Rmax = 30, dz = 0.2, Delta = 0.,Nrepeat=10,Rad=1.75,core = 0.5):
    """
    Calcul répété de l'élément de matrice <ki|T(Ei)|ki> pour différents nombre d'atomes
    
    Parameters
    ----------
    
    Nat     : nombre d'atomes pour l'expérience (vecteur)
    dz      : epaisseur du gaz (tirage uniforme) (defaut 0.2um)
    Delta   : désaccord de la sonde en unités de Gamma/2
    Nrepeat : nombre de répétition du calcul de la transmission
    Rad     : rayon du disque dans lequel les positions sont tirees (defaut 1.75um)
    core    : taille de la "sphere dure" de chaque atome (pas encore implemente)
    
    Returns
    -------
    
    imagRen : moyenne de la partie imaginaire de l'élément de matrice en fonction du nombre d'atomes (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisée par le nombre total d'atomes
    imagRenStd : écart-type de la partie imaginaire de l'élément de matrice en fonction du nombre d'atomes (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisé par le nombre total d'atomes
    realRen : moyenne de la partie réelle de l'élément de matrice en fonction du nombre d'atomes (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisée par le nombre total d'atomes
    realRenStd : écart-type de la partie réelle de l'élément de matrice en fonction du nombre d'atomes (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisé par le nombre total d'atomes
    
    """
    nexp = Nat.size
    imagRen = np.zeros((nexp,))
    imagRenStd = np.zeros((nexp,))
    realRen = np.zeros((nexp,))
    realRenStd = np.zeros((nexp,))
    
    for i in range(nexp):
        print 'Nat = ' + str(Nat[i]) + '======================================'
        trans,transStd,OD,ODstd,phi,phistd = transmision(Nat[i],Nrepeat,Rad,dz,core,Delta )
        imagRen[i] = OD;
        imagRenStd[i] = ODstd;
        realRen[i] = phi;
        realRenStd[i] = phistd;
        
    n = Nat/(np.pi*Rad**2)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.errorbar(n,imagRen,imagRenStd,marker='.')
    plt.subplot(2,1,2)
    plt.errorbar(n,realRen,realRenStd,marker='.')
    
    return imagRen,imagRenStd,realRen,realRenStd
    
# Delta in units of Gamma/2
def doResonancesVaryN(Nat=np.array([10,50,75,100,250]), dz = 0.2, Delta = np.linspace(-25,25,51),Nrepeat=3,Rad=1.75,core = 0.01,avoid=True,MethodZ = 1):
    """
    Calcul répété de l'élément de matrice <ki|T(Ei)|ki> pour différents nombre d'atomes et différents désaccords
    
    Parameters
    ----------
    
    Nat     : nombre d'atomes pour l'expérience (vecteur)
    dz      : epaisseur du gaz (tirage uniforme) (defaut 0.2um)
    Delta   : désaccord de la sonde en unités de Gamma/2 (vecteur)
    Nrepeat : nombre de répétition du calcul de la transmission
    Rad     : rayon du disque dans lequel les positions sont tirees (defaut 1.75um)
    core    : taille de la "sphere dure" de chaque atome (pas encore implemente)
    
    Returns
    -------
    
    imagRen : moyenne de la partie imaginaire de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisée par le nombre total d'atomes
    imagRenStd : écart-type de la partie imaginaire de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisé par le nombre total d'atomes
    realRen : moyenne de la partie réelle de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisée par le nombre total d'atomes
    realRenStd : écart-type de la partie réelle de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisé par le nombre total d'atomes
    
    """
    nNat = Nat.size
    nDelta = Delta.size
    imagRen = np.zeros((nNat,nDelta))
    imagRenStd = np.zeros((nNat,nDelta))
    realRen = np.zeros((nNat,nDelta))
    realRenStd = np.zeros((nNat,nDelta))
    OD0 = np.zeros((nNat,nDelta))
    
    for i in range(nNat):
        for j in range(nDelta):
            OD0[i,j] = Nat[i]/(np.pi*Rad**2) * 3*0.78**2/(2*np.pi)/(1+Delta[j]**2)
            print 'Nat = ' + str(Nat[i]) +  ' ; Delta = ' + str(Delta[j])  +  ' ; OD0 = ' + str(OD0[i,j]) + '======================================'
            trans,transStd,OD,ODstd,phi,phistd  = transmision(Nat[i],Nrepeat,Rad,dz,core,Delta[j] ,avoid,MethodZ)
            imagRen[i,j] = OD;
            imagRenStd[i,j] = ODstd;
            realRen[i,j] = phi;
            realRenStd[i,j] = phistd;
        
    n = Nat/(np.pi*Rad**2)
    
    plt.figure()
    c= plt.get_cmap('cool')

    plt.subplot(2,1,1)
    for i in range(nNat):
        plt.errorbar(Delta,imagRen[i,:],imagRenStd[i,:],marker='.',color = c(1./(nNat-1) * i))
    for i in range(nNat):
        plt.plot(Delta,OD0[i,:],'--+',color = c(1./(nNat-1) * i))
    
    plt.legend([str(s) for s in n ])
    plt.subplot(2,1,2)
    for i in range(nNat):
        plt.errorbar(Delta,realRen[i,:],realRenStd[i,:],marker='.',color = c(1./(nNat-1) * i))
    plt.legend([str(s) for s in n ])
        
    return imagRen,imagRenStd,realRen,realRenStd, n,nNat,Delta,OD0
    
def replotResonances(imagRen,imagRenStd,realRen,realRenStd, n,nNat,Delta,OD0):
    
    plt.figure()
    c= plt.get_cmap('cool')

    plt.subplot(2,1,1)
    for i in range(nNat):
        plt.errorbar(Delta,imagRen[i,:],imagRenStd[i,:],marker='.',color = c(1./(nNat-1) * i))
    for i in range(nNat):
        plt.plot(Delta,OD0[i,:],'--+',color = c(1./(nNat-1) * i))
    
    plt.legend([str(s) for s in n ])
    plt.subplot(2,1,2)
    for i in range(nNat):
        plt.errorbar(Delta,realRen[i,:],realRenStd[i,:],marker='.',color = c(1./(nNat-1) * i))
    plt.legend([str(s) for s in n ])
        
    
#    plt.figure()
#    plt.subplot(2,1,1)
#    for i in range(nNat):
#        plt.errorbar(Delta,imagRen[i,:],imagRenStd[i,:],marker='.')
#    plt.legend([str(s) for s in n ])
#    plt.subplot(2,1,2)
#    for i in range(nNat):
#        plt.errorbar(Delta,realRen[i,:],realRenStd[i,:],marker='.')
#    plt.legend([str(s) for s in n ])
    return
    

    
# Delta in units of Gamma/2
def doResonancesVaryOD(Nat=1000,OD0s=np.array([0.1,1.,2.,4.,8.]), dz = 0., Delta = np.linspace(-25,25,51),Nrepeat=3,core = 0.01,avoid=True,MethodZ = 1):
    """
    Calcul répété de l'élément de matrice <ki|T(Ei)|ki> pour différents nombre d'atomes et différents désaccords
    
    Parameters
    ----------
    
    Nat     : nombre d'atomes pour l'expérience (vecteur)
    dz      : epaisseur du gaz (tirage uniforme) (defaut 0.2um)
    Delta   : désaccord de la sonde en unités de Gamma/2 (vecteur)
    Nrepeat : nombre de répétition du calcul de la transmission
    Rad     : rayon du disque dans lequel les positions sont tirees (defaut 1.75um)
    core    : taille de la "sphere dure" de chaque atome (pas encore implemente)
    
    Returns
    -------
    
    imagRen : moyenne de la partie imaginaire de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisée par le nombre total d'atomes
    imagRenStd : écart-type de la partie imaginaire de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisé par le nombre total d'atomes
    realRen : moyenne de la partie réelle de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisée par le nombre total d'atomes
    realRenStd : écart-type de la partie réelle de l'élément de matrice en fonction du nombre d'atomes et du désaccord (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2)) divisé par le nombre total d'atomes
    
    """
    nOD = OD0s.size
    nDelta = Delta.size
    ODs = np.zeros((nOD,nDelta))
    ODsStd = np.zeros((nOD,nDelta))
    Phis = np.zeros((nOD,nDelta))
    PhisStd = np.zeros((nOD,nDelta))
    OD0 = np.zeros((nOD,nDelta))
    
    a = localtime()
    s1 = str(a.tm_hour) + 'h' + str(a.tm_min) + 'mn' + str(a.tm_sec) + 's'
    print s1
    for i in range(nOD):
        for j in range(nDelta):
            Rad = 0.78/(2*np.pi) * np.sqrt(Nat*6./OD0s[i])
            OD0[i,j] = Nat/(np.pi*Rad**2) * 3*0.78**2/(2*np.pi)/(1+Delta[j]**2)
            print 'OD = ' + str(OD0s[i]) +  ' ; Delta = ' + str(Delta[j])  +  ' ; OD0 = ' + str(OD0[i,j]) + '======================================'
            trans,transStd,OD,ODstd,phi,phistd  = transmision(Nat,Nrepeat,Rad,dz,core,Delta[j] )
            ODs[i,j] = OD;
            ODsStd[i,j] = ODstd;
            Phis[i,j] = phi;
            PhisStd[i,j] = phistd;
        
    a = localtime()
    s2 = str(a.tm_hour) + 'h' + str(a.tm_min) + 'mn' + str(a.tm_sec) + 's'
    print s1
    print s2
    
    n = Nat/(np.pi*Rad**2)
    
    plt.figure()
    c= plt.get_cmap('cool')

    plt.subplot(2,1,1)
    for i in range(nOD):
        plt.errorbar(Delta,ODs[i,:],ODsStd[i,:],marker='.',color = c(1./(nOD-1) * i))
    for i in range(nOD):
        plt.plot(Delta,OD0[i,:],'--+',color = c(1./(nOD-1) * i))
    
    plt.legend([str(s) for s in OD0s ])
    plt.subplot(2,1,2)
    for i in range(nOD):
        plt.errorbar(Delta,Phis[i,:],PhisStd[i,:],marker='.',color = c(1./(nOD-1) * i))
    plt.legend([str(s) for s in OD0s ])
        
    tag = nametag()
    name = 'Results' + os.sep + 'doResonancesVaryOD_' + tag 
    
    
    dictForMatlab = {'ODs':ODs, 'ODsStd':ODsStd, 'Phis':Phis, 'PhisStd':PhisStd, 'OD0s':OD0s, 'nOD':nOD,'Delta':Delta, 'core':core, 'avoid':avoid, 'MethodZ':MethodZ,'n':n,'OD0':OD0,'Nat':Nat,'dz':dz}
    np.savez(name + '.npz',**dictForMatlab)
    savemat(name + '.mat',dictForMatlab,oned_as='column')
    print 'saved (Matlab) to ' + name + '.mat'
    print 'saved (python) to ' + name + '.npz'
    
    return ODs,ODsStd,Phis,PhisStd, OD0s,nOD,Delta,OD0,n,Nat,OD0s,dz,Delta,core,avoid,MethodZ
    
#def replotResonances(imagRen,imagRenStd,realRen,realRenStd, n,nNat,Delta):
#    
#    plt.figure()
#    c= plt.get_cmap('cool')
#
#    plt.subplot(2,1,1)
#    for i in range(nNat):
#        plt.errorbar(Delta,imagRen[i,:],imagRenStd[i,:],marker='.',color = c(1./(nNat-1) * i))
#    for i in range(nNat):
#        plt.plot(Delta,OD0[i,:],'--+',color = c(1./(nNat-1) * i))
#    
#    plt.legend([str(s) for s in n ])
#    plt.subplot(2,1,2)
#    for i in range(nNat):
#        plt.errorbar(Delta,realRen[i,:],realRenStd[i,:],marker='.',color = c(1./(nNat-1) * i))
#    plt.legend([str(s) for s in n ])
#        
#    
##    plt.figure()
##    plt.subplot(2,1,1)
##    for i in range(nNat):
##        plt.errorbar(Delta,imagRen[i,:],imagRenStd[i,:],marker='.')
##    plt.legend([str(s) for s in n ])
##    plt.subplot(2,1,2)
##    for i in range(nNat):
##        plt.errorbar(Delta,realRen[i,:],realRenStd[i,:],marker='.')
##    plt.legend([str(s) for s in n ])
#    return
    
    
def manyExpVaryNInhomogeneousLighting(Nat=250,Rmin = 0.25,Rmax = 1., dz = 0.2, Delta = 0.,Nrepeat=5,Rad=1.75*np.sqrt(np.array([0.5,1,2,5,10])),core = 0.5):
    """
    
    """
#    nexp = Nat.size
    nexp = Rad.size
    excIn = np.zeros((nexp,))
    excInStd = np.zeros((nexp,))
    excOut = np.zeros((nexp,))
    excOutStd = np.zeros((nexp,))
    
    for i in range(nexp):
#        print 'Nat = ' + str(Nat[i]) + '======================================'
        print 'Rad = ' + str(Rad[i]) + '======================================'
        excinall  = np.zeros((Nrepeat,))
        excoutall = np.zeros((Nrepeat,))
        plt.figure()
        plt.ion()
        for j in range(Nrepeat):
            print str(j+1) + '/' + str(Nrepeat)
#            R = drawPositions(Nat[i],Rad,dz,core)
            R = drawPositions(Nat,Rad[i],dz,core)
            H = Heff(R,Delta)
            distToCenter = np.sqrt(R[:,0]*R[:,0] + R[:,1]*R[:,1])
            atomsIn = distToCenter< Rmin*Rad[i]
            atomsOut = [not a for a in atomsIn]
            AtomsIn = np.kron(np.asarray(atomsIn)*1.,[1,0,0])
            AtomsOut = np.kron(np.asarray(atomsOut)*1.,[1,0,0])
            aIn = excVectorLocalized(R,Rmax = Rmin*Rad[i])
            aOut = solve(H,aIn)
            V0 = np.abs(aOut**2)
            maxV = np.max(V0)
            V1 = 1-V0[AtomsIn==1]/maxV
            V2 = 1-V0[AtomsOut==1]/maxV
            c1 = np.transpose(np.tile(V1,(3,1)))
            c2 = np.transpose(np.tile(V2,(3,1)))
            Xin = R[atomsIn,0]/Rad[i]
            Yin = R[atomsIn,1]/Rad[i]
            Xout = R[atomsOut,0]/Rad[i]
            Yout = R[atomsOut,1]/Rad[i]
#            print np.sum(atomsIn)
#            print np.sum(atomsOut)
            plt.scatter(Xin,Yin,c = c1,marker = ">")
            plt.scatter(Xout,Yout,c = c2,marker = "<")
            plt.draw()
            excinall[j]  = np.sum(np.abs(aOut**2)*AtomsIn)
            excoutall[j] = np.sum(np.abs(aOut**2)*AtomsOut)
#        excIn[i] = np.mean(excinall)/Nat[i];
#        excInStd[i] = np.std(excinall)/Nat[i];
#        excOut[i] = np.mean(excoutall)/Nat[i];
#        excOutStd[i] = np.std(excoutall)/Nat[i];
        
        excIn[i] = np.mean(excinall)/Nat;
        excInStd[i] = np.std(excinall)/Nat;
        excOut[i] = np.mean(excoutall)/Nat;
        excOutStd[i] = np.std(excoutall)/Nat;
        
    n = Nat/(np.pi*Rad**2)
    
    plt.figure()
    plt.errorbar(n,excIn,excInStd,marker='.')
    plt.errorbar(n,excOut,excOutStd,marker='.')
    plt.legend(['in','out'])
    
    return excIn,excInStd,excOut,excOutStd,aOut, Xin, Yin, Xout, Yout, c1,c2

    
def transmisionInhomogeneous(Nat=1000,Nrepeat=10,RadExc= 0.5,stepCollect = 8,Rad=1.75,dz = 0.2,core = 0.01,Delta = 0.,avoid=True,MethodZ = 1):
    """
    Calcul de l'élément de matrice <ki|T(Ei)|ki> pour un éclairement uniforme
    
    Parameters
    ----------
    
    Nat     : nombre d'atomes pour l'expérience
    Nrepeat : nombre de répétition du calcul de la transmission
    Rad     : rayon du disque dans lequel les positions sont tirees (defaut 1.75um)
    dz   : epaisseur du gaz (tirage uniforme) (defaut 0.2um)
    core : taille de la "sphere dure" de chaque atome (pas encore implemente)
    Delta : désaccord de la sonde en unités de Gamma/2
    
    Returns
    -------
    
    mr : moyenne de la partie réelle de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    mi : moyenne de la partie imaginaire de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    sr : écart-type de la partie réelle de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    si : écart-type de la partie imaginaire de l'élément de matrice sur toutes les réalisations (en unités de (hbar OmegaRabi)^2/(hbar Gamma/2))
    
    """
    rads = np.linspace(RadExc,Rad,stepCollect+1)
    OD0 = Nat/(np.pi*Rad**2) * 3*0.78**2/(2*np.pi)
    Tim = np.zeros((Nrepeat,stepCollect,3),dtype=np.complex64)
    for i in range(Nrepeat):
        print str(i+1) + '/' + str(Nrepeat)
        R = drawPositions(Nat,Rad,dz,core)
        aExc = excVectorLocalized(R,Rmin = 0., Rmax = RadExc,pol = np.array([1. + 0j,0,0]))
        all_ax = []
        all_ay = []
        all_az = []
        for iStep in range (stepCollect):
            ax = excVectorLocalized(R,Rmin = rads[iStep], Rmax = rads[iStep+1],pol = np.array([1. + 0j,0,0]))
            ay = excVectorLocalized(R,Rmin = rads[iStep], Rmax = rads[iStep+1],pol = np.array([0,1. + 0j,0]))
            az = excVectorLocalized(R,Rmin = rads[iStep], Rmax = rads[iStep+1],pol = np.array([0,0,1. + 0j]))
            all_ax.append(ax)
            all_ay.append(ay)
            all_az.append(az)
        H = Heff(R,Delta)
        bx = solve(H,aExc)
        for iStep in range (stepCollect):
            Tim[i,iStep,0] = -0.5*1j*OD0*all_ax[iStep].conj().dot(bx)/Nat
            Tim[i,iStep,1] = -0.5*1j*OD0*all_ay[iStep].conj().dot(bx)/Nat
            Tim[i,iStep,2] = -0.5*1j*OD0*all_az[iStep].conj().dot(bx)/Nat
        
    trans = np.mean(Tim,0)
    transStd = np.std(Tim,0)
    transAbs = np.mean(np.abs(Tim),0)
    transAbsStd = np.std(np.abs(Tim),0)
        
    return trans,transStd, rads,Tim, transAbs, transAbsStd
    
    
def nametag():
    tag = ''
    a = localtime()
    if a.tm_mday<10:
        tag = tag + '0' + str(a.tm_mday) +'-'
    else:
        tag = tag + str(a.tm_mday) +'-'
    if a.tm_mon<10:
        tag = tag + '0' + str(a.tm_mon) +'-'
    else:
        tag = tag + str(a.tm_mon) +'-'
    tag = tag + str(a.tm_year) +'_'
    if a.tm_hour<10:
        tag = tag + '0' + str(a.tm_hour) +'h'
    else:
        tag = tag + str(a.tm_hour) +'h'
    if a.tm_min<10:
        tag = tag + '0' + str(a.tm_min) 
    else:
        tag = tag + str(a.tm_min) 

    return tag