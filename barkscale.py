# -*- coding: utf-8 -*-
# @Author: Aswin Sivaraman
# @Date:   2018-02-07 13:08:36
# @Last Modified by:   Aswin Sivaraman
# @Last Modified time: 2018-02-08 01:39:25

"""
The Bark scale is a psychoacoustical scale proposed by Eberhard Zwicker in 1961.
It is named after Heinrich Barkhausen who proposed the first subjective
measurements of loudness. The scale ranges from 1 to 24 and corresponds to the
first 24 critical bands of hearing.
"""

__author__ = 'Aswin Sivaraman'

class RangeDict(dict):
    def __getitem__(self, item):
        if type(item) != range:
            for key in self:
                if item in key:
                    return self[key]
        else:
            return super().__getitem__(item)

_z = RangeDict({
    range(20,99): 1,
    range(100,199): 2,
    range(200,299): 3,
    range(300,399): 4,
    range(400,509): 5,
    range(510,629): 6,
    range(630,769): 7,
    range(770,919): 8,
    range(920,1079): 9,
    range(1080,1269): 10,
    range(1270,1479): 11,
    range(1480,1719): 12,
    range(1720,1999): 13,
    range(2000,2319): 14,
    range(2320,2699): 15,
    range(2700,3149): 16,
    range(3150,3699): 17,
    range(3700,4399): 18,
    range(4400,5299): 19,
    range(5300,6399): 20,
    range(6400,7699): 21,
    range(7700,9499): 22,
    range(9500,11999): 23,
    range(12000,15499): 24
})

_zlower = {
    1: 20,
    2: 100,
    3: 200,
    4: 300,
    5: 400,
    6: 510,
    7: 630,
    8: 770,
    9: 920,
    10: 1080,
    11: 1270,
    12: 1480,
    13: 1720,
    14: 2000,
    15: 2320,
    16: 2700,
    17: 3150,
    18: 3700,
    19: 4400,
    20: 5300,
    21: 6400,
    22: 7700,
    23: 9500,
    24: 12000
}

_zupper = {
    1: 99,
    2: 199,
    3: 299,
    4: 399,
    5: 509,
    6: 629,
    7: 769,
    8: 919,
    9: 1079,
    10: 1269,
    11: 1479,
    12: 1719,
    13: 1999,
    14: 2319,
    15: 2699,
    16: 3149,
    17: 3699,
    18: 4399,
    19: 5299,
    20: 6399,
    21: 7699,
    22: 9499,
    23: 11999,
    24: 15499
}

def hz2z(hz):
	return _z[hz]

def z2hz(z, boundary="lower"):
    if boundary is "upper":
        return _zupper[z]
    else:
        return _zlower[z]

def z2hz_lower(z):
    return _zlower[z]

def z2hz_upper(z):
    return _zupper[z]

def hz2bin(hz, sr=44100, fft_size=512):
    return int(hz*fft_size/sr)

def bin2hz(k, sr=44100, fft_size=512):
    return int(k*sr/fft_size)

def bin2z(k, sr=44100, fft_size=512):
    return hz2z(bin2hz(k, sr, fft_size))
