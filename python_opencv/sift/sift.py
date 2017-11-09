# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 09:31:47 2016

@author: Eddy_Zheng
"""

import cv2

# 读取图像
im = cv2.imread('mypython/python_ml/python_opencv/sift/1.jpg')
cv2.imshow('original',im)
#cv2.waitKey()

# 下采样
#im_lowers = cv2.pyrDown(im) 
#cv2.imshow('im_lowers',im_lowers)

# 检测特征点
#s = cv2.SIFT() # 调用SIFT
s = cv2.SURF() # 调用SURF
keypoints = s.detect(im)

# 显示特征点
for k in keypoints:
    cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),1,(0,255,0),-1)
    #cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)


cv2.imshow('SURF_features',im)
cv2.waitKey()
cv2.destroyAllWindows()
