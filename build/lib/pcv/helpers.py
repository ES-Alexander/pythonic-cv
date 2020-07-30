#!/usr/bin/env python3

import cv2

def downsize(img, ratio):
    ''' downsize 'img' by 'ratio'. '''
    return cv2.resize(img,
                      tuple(dim // ratio for dim in reversed(img.shape[:2])),
                      interpolation = cv2.INTER_AREA)

def channel_options(img, rank=False):
    ''' Create a composite image of img in all of opencv's colour channels

    |img| -> | blue       | green      | red         |
             | hue        | saturation | value       |
             | hue2       | luminosity | saturation2 |
             | lightness  | green-red  | blue-yellow |
             | lightness2 | u          | v           |

    'rank' is a boolean? specifying whether to also return a ranking of each
        channel by variability/sharpness/contrast/other? !NOT YET IMPLEMENTED!
        TODO
        -> make a string maybe, with several options available, or select
            multiple options in a list and get back an array or dataframe or
            something
        -> important to make nicely stackable to run on video and determine
           statistics on the best option for a given use case

    '''
    B,G,R = cv2.split(img)
    H,S,V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    H2,L2,S2 = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
    L,a,b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    L3,u,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LUV))
    channels = (((B, 'blue'), (G, 'green'), (R, 'red')),
                ((H, 'hue'), (S, 'saturation'), (V, 'value')),
                ((H2, 'hue2'), (L2, 'luminosity'), (S2, 'saturation2')),
                ((L, 'lightness'), (a, 'green-red'), (b, 'blue-yellow')),
                ((L3,'lightness2'), (u, 'u'), (v, 'v')))
    out = []
    for row in channels:
        img_row = []
        for img, name in row:
            cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, 255, 1)
            img_row.append(img)
        out.append(cv2.hconcat(img_row))
    return cv2.vconcat(out)
