#!/usr/bin/env python3

# @Author: Aswin Sivaraman
# @Email: aswin.sivaraman@gmail.com
# @Last Modified by:   Aswin Sivaraman
# @Last Modified time: 2018-02-14 16:52:46

__description__ = 'Applies a psychoacoustic model to a WAV file.'

import argparse
import numpy as np
import numpy.ma as ma
import time
from math import cos, sin, pi, floor
from barkscale import *
from tqdm import tqdm
from scipy.io.wavfile import read
from scipy.stats import gmean

def MPEG1(filepath, fft_size, overlap, save=False, plot=False, step=0):
    "Perceptual Model: ISO 11172-3 (MPEG-1) Psychoacoustic Model 1"

    # Load in the input file and normalize it
    sample_rate, data = read(filepath)
    data = np.divide(data, max(abs(data.min()), abs(data.max())))
    # data = np.append(np.zeros([1,fft_size]), data)
    # data = np.append(data, np.zeros([1,fft_size*2-(len(data)%fft_size)]))
    window = np.hanning(fft_size)
    hop = floor(fft_size * (1-overlap))
    frames = 1 + floor((len(data)-fft_size)/hop)

    # ======================================================= #
    ## Step 1 — Spectral Analysis and SPL Normalization
    # ======================================================= #

    if step > 0:
        Y = np.load(filepath[:-4]+'-Y.npy')
        P = np.load(filepath[:-4]+'-P.npy')
    else:
        # Create a DFT matrix (F)
        i, j = np.meshgrid(np.arange(fft_size), np.arange(fft_size))
        omega = np.exp(-2J*pi/fft_size)
        F = np.power(omega,i*j)

        # Construct a data matrix (X) using a Hann window and hop-length
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
        PN = 96 # Power normalization term
        P = 10*ma.log10(np.square(np.abs(Y))).filled(10**-10)
        P += (PN - P.max())
        if save:
            np.save(filepath[:-4]+'-P.npy', P)
        print(">> Estimated the power spectral density")

    # ======================================================= #
    ## Step 2 — Identification of Tonal and Noise Maskers
    # ======================================================= #

    num_bins = P.shape[0]//2+1
    num_frames = P.shape[1]
    num_criticalbands = hz2z(sample_rate//2)-1

    plot_specific_frame = 100

    if step > 1:
        S_T = np.load(filepath[:-4]+'-S_T.npy')
        P_TM = np.load(filepath[:-4]+'-P_TM.npy')
        P_NM = np.load(filepath[:-4]+'-P_NM.npy')
    else:
        # Define tonal neighborhoods (PAINTER, Eq. 22)
        neighborhoods = RangeDict({
            range(2,352): 2,
            range(352,512): 3
            # range(int(127),int(250)): 6
        })

        # Find the tonal set (S_T) (PAINTER, Eq. 21)
        # Compute the tonal maskers (P_TM) (PAINTER, Eq. 23)
        # Compute the noise makers for each critical band (P_NM) (PAINTER, Eq. 24)
        S_T = np.zeros_like(P) != 0
        S_TE = np.zeros_like(P) != 0
        P_TM = np.zeros_like(P)
        P_NM = np.zeros_like(P)
        pbar = tqdm(total=(num_frames),unit='frames')
        for f in range(num_frames):
            pbar.update()
            for k in range(2,512):
                _tonal = P[k-1,f] < P[k,f] > P[k+1,f]
                if _tonal:
                    for n in range(2,neighborhoods[k]+1):
                        _tonal = _tonal and (P[k-n,f]+7 < P[k,f] > P[k+n,f]+7)
                S_T[k,f] = _tonal
                if _tonal:
                    S_TE[k,f] = S_TE[k-1,f] = S_TE[k+1,f] = _tonal
                    for n in range(2,neighborhoods[k]+1):
                        S_TE[k-n,f] = S_TE[k+n,f] = _tonal
                    P_TM[k,f] = 10 * np.log10(
                            np.power(10, 0.1*P[k-1,f]) +
                            np.power(10, 0.1*P[k,f]) +
                            np.power(10, 0.1*P[k+1,f])
                        )
            _P = P * (S_TE*-1 + 1)
            for z in range(1,num_criticalbands+1):
                _l = hz2bin(z2hz(z, "lower"), sr=sample_rate, fft_size=fft_size)
                _u = hz2bin(z2hz(z, "upper"), sr=sample_rate, fft_size=fft_size)
                _kbar = int(gmean(range(_l,_u))) # (PAINTER, Eq. 25)
                # _j = []
                # for j in range(_l,_u+1):
                #     _nontonal = not (S_T[j,f] or S_T[j-1,f] or S_T[j+1,f])
                #     if neighborhoods[_kbar]:
                #         for n in range(2,neighborhoods[_kbar]+1):
                #             _nontonal = _nontonal and not (S_T[j-n,f] or S_T[j+n,f])
                #         if _nontonal:
                #             _j.append(j)
                # P_NM[_kbar,f] = 10 * np.log10(1 + sum([np.power(10, 0.1*_P[j,f]) for j in _j]))
                P_NM[_kbar,f] = 10 * np.log10(sum(np.power(10, 0.1*_P[_l:_u,f])))
        pbar.close()
        if save:
            np.save(filepath[:-4]+'-S_T.npy', S_T[:num_bins])
            np.save(filepath[:-4]+'-S_TE.npy', S_TE[:num_bins])
            np.save(filepath[:-4]+'-P_TM.npy', P_TM[:num_bins])
            np.save(filepath[:-4]+'-P_NM.npy', P_NM[:num_bins])
        print(">> Found tonal set and tonal neighbors")
        print(">> Computed tonal maskers")
        print(">> Computed noise makers for each critical band")

    if 0: #plot:
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(24,9), dpi=100)
        axes[0,0].imshow(abs(Y[:num_bins]),origin='lower', aspect='auto')
        axes[0,1].imshow(abs(P[:num_bins]),origin='lower', aspect='auto')
        axes[0,2].imshow(abs(S_T[:num_bins]),origin='lower', aspect='auto')
        axes[1,0].imshow(abs(S_TE[:num_bins]),origin='lower', aspect='auto')
        axes[1,1].imshow(abs(P_TM[:num_bins]),origin='lower', aspect='auto')
        axes[1,2].imshow(abs(P_NM[:num_bins]),origin='lower', aspect='auto')
        axes[0,0].set_title(r'Short-Time Fourier Transform, $Y$')
        axes[0,1].set_title(r'Power Spectral Density, $P$')
        axes[0,2].set_title(r'Tonal Set, $S_T$')
        axes[1,0].set_title(r'Tonal Neighbors, $S_{TE}$')
        axes[1,1].set_title(r'Tonal Maskers, $P_{TM}$')
        axes[1,2].set_title(r'Noise Maskers, $P_{NM}$')
        axes[0,0].axis('off')
        axes[0,1].axis('off')
        axes[0,2].axis('off')
        axes[1,0].axis('off')
        axes[1,1].axis('off')
        axes[1,2].axis('off')
        # fig.text(0.5, 0.005, 'Time Frame [f]', ha='center')
        # fig.text(0.005, 0.5, 'Frequency Bins [k]', va='center', rotation='vertical')
        plt.tight_layout()
        plt.savefig('plt_step2_'+str(time.time())+'.png')
        plt.show()

    # ======================================================= #
    ## Step 3 — Decimation and Reorganization of Maskers
    # ======================================================= #

    if step > 2:
        T_q = np.load(filepath[:-4]+'-T_q.npy')
    else:

        oldP_TM = P_TM.copy()
        oldP_NM = P_NM.copy()

        print(np.sum(oldP_NM > 0) + np.sum(oldP_TM > 0))

        _f, _step = np.linspace(0, sample_rate, fft_size, endpoint=False, retstep=True)
        _f = np.add(_f, _step/2)
        T_q = (3.64 * np.power(_f/1000, -0.8)) \
            - (6.5 * np.exp(-0.6*(np.power((_f/1000)-3.3, 2)))) \
            + (0.001 * np.power(_f/1000, 4))
        assert T_q.shape[0] == P.shape[0]
        if save:
            np.save(filepath[:-4]+'-T_q.npy', T_q[:num_bins])
        print(">> Constructed absolute threshold")

        if plot:
            l = [0]*8
            _pz = [bin2z(r, sr=sample_rate, fft_size=fft_size, traunmuller=True) for r in range(num_bins)]
            fig, axes = plt.subplots(2, sharex=True, sharey=True, figsize=(24,9), dpi=100)
            plt.xlabel('Bark [z]')
            plt.ylabel('Power [dB-SPL]')
            l[0], = axes[0].plot(_pz,P[:num_bins,0], label='P')
            l[1], = axes[0].plot(_pz,T_q[:num_bins], label='T_q')
            l[2], = axes[0].plot(_pz,oldP_TM[:num_bins,0], label='P_TM')
            l[3], = axes[0].plot(_pz,oldP_NM[:num_bins,0], label='P_NM')
            axes[0].legend(handles=[l[0],l[1],l[2],l[3]])
            axes[0].set_title(r'Before decimation')

        for f in range(num_frames):
            for k in range(num_bins):
                if P_TM[k,f] < T_q[k]:
                    S_T[k,f] = False
                    P_TM[k,f] = 0
                if P_NM[k,f] < T_q[k]:
                    P_NM[k,f] = 0
        print(">> Discarded tonal or noise makers below the absolute threshold")

        for f in range(num_frames):
            _zw = 0.5
            _zs = 0.25
            for z in np.arange(1,num_criticalbands,_zs):
                _bl = z2bin(z-_zw, sr=sample_rate, fft_size=fft_size, traunmuller=True)
                _bh = z2bin(z+_zw, sr=sample_rate, fft_size=fft_size, traunmuller=True)
                _localmax = max(P_TM[_bl:_bh,f].max(), P_NM[_bl:_bh,f].max())
                if _localmax > 0:
                    for k in range(_bl,_bh):
                        if P_TM[k,f] < _localmax:
                            S_T[k,f] = False
                            P_TM[k,f] = 0
                        if P_NM[k,f] < _localmax:
                            P_NM[k,f] = 0
        print(">> Consolidated maskers that are within +/- 0.5 barks")

        _T = np.zeros_like(P_TM)
        _N = np.zeros_like(P_TM)
        for f in range(num_frames):
            for k in range(num_bins):
                if 1 <= k <= 264:
                    i = k
                    _T[i,f] = P_TM[k,f]
                    _N[i,f] = P_NM[k,f]
                    P_TM[k,f] = 0
                    P_NM[k,f] = 0
                if 265 <= k <= 512:
                    i = k + (k % 2)
                    _T[i,f] = P_TM[k,f]
                    _N[i,f] = P_NM[k,f]
                    P_TM[k,f] = 0
                    P_NM[k,f] = 0
        P_TM = _T
        P_NM = _N
        print(">> Subsampled maskers")

        if plot:
            l[4], = axes[1].plot(_pz,P[:num_bins,0])
            l[5], = axes[1].plot(_pz,T_q[:num_bins])
            l[6], = axes[1].plot(_pz,P_TM[:num_bins,0])
            l[7], = axes[1].plot(_pz,P_NM[:num_bins,0])
            axes[1].set_title(r'After decimation')

            # Slider parameters
            axcolor = 'lightgoldenrodyellow'
            axframe = plt.axes([0.2, 0.05, 0.6, 0.01], facecolor=axcolor)
            sframe = Slider(axframe, 'Frame', 0, num_frames-1, valinit=0, valfmt='%0.0f')
            def update(val):
                frame = int(np.around(sframe.val))
                l[0].set_ydata(P[:num_bins,frame])
                l[2].set_ydata(oldP_TM[:num_bins,frame])
                l[3].set_ydata(oldP_NM[:num_bins,frame])
                l[4].set_ydata(P[:num_bins,frame])
                l[6].set_ydata(P_TM[:num_bins,frame])
                l[7].set_ydata(P_NM[:num_bins,frame])
            sframe.on_changed(update)
            plt.show()

    # ======================================================= #
    ## Step 4 — Calculation of Individual Masking Thresholds
    # ======================================================= #

    if step > 3:
        T_TM = np.load(filepath[:-4]+'-T_TM.npy')
        T_NM = np.load(filepath[:-4]+'-T_NM.npy')
    else:
        # Compute individual threshold contributions from each tonal
        # and noise masker (PAINTER, Eq. 30/31/32)
        T_TM = np.zeros_like(P)
        T_NM = np.zeros_like(P)
        pbar = tqdm(total=(num_frames),unit='frames')
        for f in range(num_frames):
            pbar.update()
            for j in range(num_bins):
                if S_T[j,f]: # Compute spreading function only if 'j' is a masker
                    _zj = min(24,max(1,bin2z(j, sr=sample_rate, fft_size=fft_size, traunmuller=True)))
                    _bl = z2bin(max(1,_zj-3), sr=sample_rate, fft_size=fft_size, traunmuller=True)
                    _bh = min(num_bins-1,z2bin(min(24,_zj+8), sr=sample_rate, fft_size=fft_size, traunmuller=True))
                    for i in range(_bl,_bh):
                        _zi = bin2z(i, sr=sample_rate, fft_size=fft_size, traunmuller=True)
                        _SFTM = 0
                        _SFNM = 0
                        _dz = _zi - _zj
                        if -3 <= _dz < -1:
                            _SFTM = 17*_dz - 0.4*P_TM[j,f] + 11
                            _SFNM = 17*_dz - 0.4*P_NM[j,f] + 11
                        if -1 <= _dz < 0:
                            _SFTM = (0.4*P_TM[j,f] + 6)*_dz
                            _SFNM = (0.4*P_NM[j,f] + 6)*_dz
                        if  0 <= _dz < 1:
                            _SFTM = -17*_dz
                            _SFNM = -17*_dz
                        if  1 <= _dz < 8:
                            _SFTM = (0.15*P_TM[j,f] - 17)*_dz - (0.15*P_TM[j,f])
                            _SFNM = (0.15*P_NM[j,f] - 17)*_dz - (0.15*P_NM[j,f])
                        # print('masker j={}, maskee i={}, masker zj={}, maskee zi={}, dz={}, SF={}'.format(j,i,_zj,_zi,_dz,_SFTM))
                        T_TM[i,f] += P_TM[j,f] - 0.275*_zj + _SFTM - 6.025
                        T_NM[i,f] += P_NM[j,f] - 0.175*_zj + _SFNM - 2.025
            plt.plot(P_NM[:,f])
            plt.show()
        pbar.close()
        if save:
            np.save(filepath[:-4]+'-T_TM.npy', T_TM[:num_bins])
            np.save(filepath[:-4]+'-T_NM.npy', T_NM[:num_bins])
        print(">> Computed individual tonal & noise masker thresholds")

    print(T_TM.max())
    print(T_NM.max())

    # ======================================================= #
    ## Step 5 — Calculation of Global Masking Thresholds
    # ======================================================= #

    if step > 4:
        T_g = np.load(filepath[:-4]+'-T_g.npy')
    else:
        # Combine individual masking thresholds to create a global
        # masking threshold at each frequency bin (PAINTER, Eq. 33)
        T_g = np.zeros_like(P)
        pbar = tqdm(total=(num_frames),unit='frames')
        for f in range(num_frames):
            pbar.update()
            for k in range(num_bins):
                T_g[k,f] = 10*np.log10(np.sum([
                        np.power(10, 0.1*T_q[k]),
                        np.power(10, 0.1*T_TM[k,f]),
                        np.power(10, 0.1*T_NM[k,f])
                    ]))
        pbar.close()
        if save:
            np.save(filepath[:-4]+'-T_g.npy', T_g[:num_bins])
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
            np.save(filepath[:-4]+'-W.npy', W[:num_bins])
        print(">> Generated psychoacoustic weights")
        print('-- Saved results to '+filepath[:-4]+'-W.npy, of shape: '+str(W.shape)+' min: '+str(W.min())+' max: '+str(W.max()))

    if plot:
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
        plt.savefig('plt_step6_2_'+str(time.time())+'.png')
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
        from matplotlib.widgets import Slider
        from mpl_toolkits.mplot3d import Axes3D

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
