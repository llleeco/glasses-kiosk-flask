from scipy.spatial import distance
import copy
import math
import operator

def is_warm(lab_b, a):
    '''
    파라미터 lab_b = [skin_b, hair_b, eye_b]
    a = 가중치 [skin, hair, eye]
    질의색상 lab_b값에서 warm의 lab_b, cool의 lab_b값 간의 거리를
    각각 계산하여 warm이 가까우면 1, 반대 경우 0 리턴
    '''
    # standard of skin
    warm_b_std = [11.6518]
    cool_b_std = [4.64255]

    warm_dist = 0
    cool_dist = 0

    # warm 및 cool과의 거리 계산
    warm_dist = abs(lab_b[0] - warm_b_std[0]) * a[0]
    cool_dist = abs(lab_b[0] - cool_b_std[0]) * a[0]

    if(warm_dist <= cool_dist):
        return 1 #warm
    else:
        return 0 #cool

def is_spr(hsv_s, a):
    '''
    파라미터 hsv_s = [skin_s, hair_s, eye_s]
    a = 가중치 [skin, hair, eye]
    질의색상 hsv_s값에서 spring의 hsv_s, fall의 hsv_s값 간의 거리를
    각각 계산하여 spring이 가까우면 1, 반대 경우 0 리턴
    '''
    #skin
    spr_s_std = [18.59296]
    fal_s_std = [27.13987]

  # spring 및 fall과의 거리 계산
    spr_dist = abs(hsv_s[0] - spr_s_std[0]) * a[0]
    fal_dist = abs(hsv_s[0] - fal_s_std[0]) * a[0]

    if(spr_dist <= fal_dist):
        return 1 #spring
    else:
        return 0 #fall

def is_smr(hsv_s, a):
    '''
    파라미터 hsv_s = [skin_s, hair_s, eye_s]
    a = 가중치 [skin, hair, eye]
    질의색상 hsv_s값에서 summer의 hsv_s, winter의 hsv_s값 간의 거리를
    각각 계산하여 summer가 가까우면 1, 반대 경우 0 리턴
    '''
    #skin
    smr_s_std = [12.5]
    wnt_s_std = [16.73913]

    # summer 및 winter와의 거리 계산
    smr_dist = abs(hsv_s[0] - smr_s_std[0]) * a[0]
    wnt_dist = abs(hsv_s[0] - wnt_s_std[0]) * a[0]

    if(smr_dist <= wnt_dist):
        return 1 #summer
    else:
        return 0 #winter