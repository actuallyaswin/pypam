#!/usr/bin/env python3

# @Author: Aswin Sivaraman
# @Email: aswin.sivaraman@gmail.com
# @Last Modified by:   Aswin Sivaraman
# @Last Modified time: 2018-02-08 02:48:58

__description__ = 'Applies a psychoacoustic model to a WAV file.'

import argparse
import numpy as np
import numpy.ma as ma
import time
from barkscale import *
from tqdm import tqdm
from scipy.io.wavfile import read
from scipy.stats import gmean

def MPEG1(filepath, fft_size, overlap, save=False, plot=False, step=0):
    "Perceptual Model: ISO 11172-3 (MPEG-1) Psychoacoustic Model 1"

    # Load in the input file and normalize it
    sample_rate, data = read(filepath)
    data = np.divide(data, max(abs(data.min()), abs(data.max())))

    # ======================================================= #
    ## Step 1 — Spectral Analysis and SPL Normalization
    # ======================================================= #

    if step > 0:
        Y = np.load(filepath[:-4]+'-Y.npy')
        P = np.load(filepath[:-4]+'-P.npy')
    else:
        # Create a DFT matrix (F)
        from math import cos, sin, pi, floor
        i, j = np.meshgrid(np.arange(fft_size), np.arange(fft_size))
        omega = np.exp(-2J*pi/fft_size)
        F = np.power(omega,i*j)

        # Construct a data matrix (X) using a Hann window and hop-length
        window = np.hanning(fft_size)
        hop = floor(fft_size * (1-overlap))
        frames = 1 + floor((len(data)-fft_size)/hop)
        X = []
        for index in range(frames):
            frame = data[index*hop : index*hop+fft_size]
            X.append(frame*window)
        X = np.divide(np.array(X),2)

        # Apply the DFT matrix (F) to the data matrix (X) to get the STFT matrix (Y)
        Y = np.dot(X,F).T
        if save:
            np.save(filepath[:-4]+'-Y.npy', Y)

        # Estimate the power spectral density (P)
        PN = 90.302 # Power normalization term
        P = PN + 10*ma.log10(np.square(np.abs(Y))).filled(10**-10)
        if save:
            np.save(filepath[:-4]+'-P.npy', P)
        print(">> Estimated the power spectral density")

    # ======================================================= #
    ## Step 2 — Identification of Tonal and Noise Maskers
    # ======================================================= #

    num_bins = P.shape[0]//2+1
    num_frames = P.shape[1]
    num_criticalbands = 24

    if step > 1:
        S_T = np.load(filepath[:-4]+'-S_T.npy')
        P_TM = np.load(filepath[:-4]+'-P_TM.npy')
        P_NM = np.load(filepath[:-4]+'-P_NM.npy')
    else:
        # Define tonal neighborhoods (PAINTER, Eq. 22)
        neighborhoods = RangeDict({
            range(                    0, int(fft_size*63/512)): 2,
            range( int(fft_size*63/512),int(fft_size*127/512)): 3,
            range(int(fft_size*127/512),int(fft_size))        : 6
        })

        # Find the tonal set (S_T) (PAINTER, Eq. 21)
        S_T = np.zeros_like(P) != 0
        for f in range(num_frames):
            for k in range(1,num_bins-1):
                _tonal = P[k-1,f] < P[k,f] > P[k+1,f]
                if _tonal:
                    for n in range(2,neighborhoods[k]+1):
                        _tonal = _tonal and (P[k-n,f]+7 < P[k,f] > P[k+n,f]+7)
                S_T[k,f] = _tonal
        if save:
            np.save(filepath[:-4]+'-S_T.npy', S_T)
        print(">> Found tonal set")

        # Compute the tonal maskers (P_TM) (PAINTER, Eq. 23)
        P_TM = np.zeros_like(P)
        for f in range(num_frames):
            for k in range(num_bins):
                if S_T[k,f]:
                    P_TM[k,f] = 10 * np.log10(
                            10**(0.1*P[k-1,f]) +
                            10**(0.1*P[k,f]) +
                            10**(0.1*P[k+1,f])
                        )
        if save:
            np.save(filepath[:-4]+'-P_TM.npy', P_TM)
        print(">> Computed tonal maskers")

        # Compute the noise makers for each critical band (P_NM) (PAINTER, Eq. 24)
        P_NM = np.zeros_like(P)
        for f in range(num_frames):
            for z in range(1,num_criticalbands+1):
                _l = hz2bin(z2hz(z, "lower"), sr=sample_rate, fft_size=fft_size)
                _u = hz2bin(z2hz(z, "upper"), sr=sample_rate, fft_size=fft_size)
                _kbar = int(gmean(range(_l,_u+1))) # (PAINTER, Eq. 25)
                _j = []
                for j in range(_l,_u):
                    n = neighborhoods[_kbar]
                    # print('z={}, _l={}, _u={}, _kbar={}, n={}'.format(z,_l,_u,_kbar,n))
                    if not (S_T[j,f] or S_T[j-n,f] or S_T[j-1,f] or S_T[j+1,f] or S_T[j+n,f]):
                        _j.append(j)
                P_NM[_kbar,f] = 10 * np.log10(1 + sum([10**(0.1*P[j,f]) for j in _j]))
        if save:
            np.save(filepath[:-4]+'-P_NM.npy', P_NM)
        print(">> Computed noise makers for each critical band")

    if plot:
        fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(24,6), dpi=100)
        axes[0].imshow(abs(Y[:num_bins]),origin='lower', aspect='auto')
        axes[1].imshow(abs(P[:num_bins]),origin='lower', aspect='auto')
        axes[2].imshow(abs(S_T[:num_bins]),origin='lower', aspect='auto')
        axes[3].imshow(abs(P_TM[:num_bins]),origin='lower', aspect='auto')
        axes[4].imshow(abs(P_NM[:num_bins]),origin='lower', aspect='auto')
        axes[0].set_title(r'Short-Time Fourier Transform, $Y$')
        axes[1].set_title(r'Power Spectral Density, $P$')
        axes[2].set_title(r'Tonal Set, $S_T$')
        axes[3].set_title(r'Tonal Maskers, $P_{TM}$')
        axes[4].set_title(r'Noise Maskers, $P_{NM}$')
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        axes[3].axis('off')
        axes[4].axis('off')
        fig.text(0.5, 0.005, 'Time Frame [f]', ha='center')
        fig.text(0.005, 0.5, 'Frequency Bins [k]', va='center', rotation='vertical')
        plt.tight_layout()
        plt.savefig('plt_step2.png')
        plt.show()

    # ======================================================= #
    ## Step 3 — Decimation and Reorganization of Maskers
    # ======================================================= #

    if step > 2:
        T_q = np.load(filepath[:-4]+'-T_q.npy')
    else:
        def absoluteThreshold(fs, N, num_frames):
            f, step = np.linspace(0, fs, N, endpoint=False, retstep=True)
            f = np.add(f, step/2)
            T = np.squeeze([ ((
                    3.64 * np.power(f/1000, -0.8))
                    - (6.5 * np.exp(-0.6*(np.power((f/1000)-3.3, 2))))
                    + (0.001 * np.power(f/1000, 4)))
                ])
            return np.tile(T, (frames, 1)).T
        T_q = absoluteThreshold(sample_rate, fft_size, num_frames)
        if save:
            np.save(filepath[:-4]+'-T_q.npy', T_q)

        # Discard tonal or noise makers below the absolute threshold (T_q)
        for f in range(num_frames):
            for k in range(num_bins):
                if P_TM[k,f] < T_q[k,f]:
                    S_T[k,f] = False
                    P_TM[k,f] = 0
                if P_NM[k,f] < T_q[k,f]:
                    P_NM[k,f] = 0

        print(">> Discarded tonal or noise makers below the absolute threshold")

    # ======================================================= #
    ## Step 4 — Calculation of Individual Masking Thresholds
    # ======================================================= #

    if step > 3:
        T_TM = np.load(filepath[:-4]+'-T_TM.npy')
        T_NM = np.load(filepath[:-4]+'-T_NM.npy')
    else:
        pbar = tqdm(total=(num_frames*num_bins*num_bins),unit='bins')

        # Compute individual threshold contributions from each tonal
        # and noise masker (PAINTER, Eq. 30/31/32)
        T_TM = np.zeros([num_frames,num_bins,num_bins])
        T_NM = np.zeros([num_frames,num_bins,num_bins])
        for f in range(num_frames):
            for j in range(num_bins):
                for i in range(num_bins):
                    _zi = bin2z(i, sr=sample_rate, fft_size=fft_size)
                    _zj = bin2z(j, sr=sample_rate, fft_size=fft_size)
                    _SF = 0
                    if _zi and _zj:

                        _dz = _zi - _zj
                        if -3 <= _dz < -1:
                            _SF = 17*_dz - 0.4*P_TM[j,f] + 11
                        if -1 <= _dz < 0:
                            _SF = (0.4*P_TM[j,f] + 6)*_dz
                        if  0 <= _dz < 1:
                            _SF = -17*_dz
                        if  1 <= _dz < 8:
                            _SF = (0.15*P_TM[j,f] - 17)*_dz - (0.15*P_TM[j,f])

                        T_TM[f,i,j] = P_TM[j,f] - 0.275*_zj + _SF - 6.025
                        T_NM[f,i,j] = P_NM[j,f] - 0.175*_zj + _SF - 2.025
                    pbar.update()
        pbar.close()
        if save:
            np.save(filepath[:-4]+'-T_TM.npy', T_TM)
            np.save(filepath[:-4]+'-T_NM.npy', T_NM)
        print(">> Computed individual tonal & noise masker thresholds")

    # ======================================================= #
    ## Step 5 — Calculation of Global Masking Thresholds
    # ======================================================= #

    if step > 4:
        T_g = np.load(filepath[:-4]+'-T_g.npy')
    else:
        # Combine individual masking thresholds to create a global
        # masking threshold at each frequency bin (PAINTER, Eq. 33)
        T_g = np.zeros_like(P)
        for f in range(num_frames):
            for i in range(num_bins):
                T_g[i,f] = 10*np.log10(np.sum([
                        np.power(10, 0.1*T_q[i,f]),
                        np.sum(np.power(10, 0.1*T_TM[f,i])),
                        np.sum(np.power(10, 0.1*T_NM[f,i]))
                    ]))
        if save:
            np.save(filepath[:-4]+'-T_g.npy', T_g)
        print(">> Computed global masking threshold")

    # ======================================================= #
    ## Step 6 — Generate Psychoacoustic Weights
    # ======================================================= #

    if step > 5:
        W = np.load(filepath[:-4]+'-W.npy')
    else:
        # Generate psychoacoustic weights
        W = np.log10( np.power(10, 0.1*P) / np.power(10, 0.1*T_g) + 1)
        if save:
            np.save(filepath[:-4]+'-W.npy', W)
        print(">> Generated psychoacoustic weights")

    if plot:
        fig, axes = plt.subplots(2, 1, sharex=True)
        l1, = axes[0].plot(P[:num_bins,30])
        l2, = axes[0].plot(T_g[:num_bins,30])
        axes[0].legend([l1,l2],[r'Power Spectral Density, $P$',r'Global Masking Threshold, $T_g$'])
        axes[0].set_xlabel('Frequency Bin')
        axes[0].set_ylabel('dB-SPL')
        axes[1].plot(W[:num_bins,30])
        axes[1].set_ylabel('Perceptual Weight Values')
        axes[1].set_xlabel('Frequency Bin')
        plt.savefig('plt_step6_1.png')
        plt.show()
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12,4), dpi=100)
        fig.colorbar( axes[0].imshow(P[:num_bins], origin="lower", aspect='auto'), ax=axes[0] )
        fig.colorbar( axes[1].imshow(T_g[:num_bins], origin="lower", aspect='auto'), ax=axes[1] )
        fig.colorbar( axes[2].imshow(W[:num_bins], origin="lower", aspect='auto'), ax=axes[2] )
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        axes[0].set_title(r'Power Spectral Density, $P$')
        axes[1].set_title(r'Global Masking Threshold, $T_g$')
        axes[2].set_title(r'Perceptual Weights, $H$')
        fig.text(0.5, 0.005, 'Time Frame [f]', ha='center')
        fig.text(0.005, 0.5, 'Frequency Bins [k]', va='center', rotation='vertical')
        plt.tight_layout()
        plt.savefig('plt_step6_2.png')
        plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('filepath', type=str,
                        help='the path to the input WAV file')
    parser.add_argument('--model', type=str, metavar='', default='MPEG1',
                        help='the name of the psychoacoustic model to apply (default: MPEG1)')
    parser.add_argument('--fft_size', metavar='', type=int, default=1024,
                        help='number of frequency bins (default: 1024)')
    parser.add_argument('--fft_overlap', metavar='', type=float, default=0.75,
                        help='percentage of overlap between frames (default: 0.75)')
    parser.add_argument('--save', action='store_true',
                        help='in order to save intermediate values')
    parser.add_argument('--plot', action='store_true',
                        help='in order to plot model')
    parser.add_argument('--step', metavar='', type=int, default=0,
                        help='in order to skip to a particular step')

    args = parser.parse_args()

    if args.model not in globals():
        print('Unable to locate model {}, now using MPEG1...'.format(args.model))
        args.model = 'MPEG1'

    if args.plot:
        import matplotlib.pyplot as plt

    # Tick
    time_start = time.time()

    # Call the desired psychoacoustic model function
    globals()[args.model](
        filepath=args.filepath,
        fft_size=args.fft_size,
        overlap=args.fft_overlap,
        save=args.save,
        plot=args.plot,
        step=args.step
    )

    # Tock
    time_end = time.time()
    print('-- Elapsed time (in seconds): '+str(time_end-time_start))
