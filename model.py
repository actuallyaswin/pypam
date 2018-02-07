# -*- coding: utf-8 -*-
# @Author: Aswin Sivaraman
# @Date:   2018-02-07 13:08:02
# @Last Modified by:   Aswin Sivaraman
# @Last Modified time: 2018-02-07 14:16:25

__description__ = 'Applies a psychoacoustic model to a WAV file.'

import argparse
import numpy as np
import numpy.ma as ma
import time
from barkscale import *


def MPEG1(filepath, fft_size, overlap, save=False, plot=False):
    "Perceptual Model: ISO 11172-3 (MPEG-1) Psychoacoustic Model 1"

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
        plot=args.plot
    )

    # Tock
    time_end = time.time()
    print('-- Elapsed time (in seconds): '+str(time_end-time_start))