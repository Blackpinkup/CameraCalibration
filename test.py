import cv2
import os
import glob
import numpy as np

#w = 13
#h = 6
#img = cv2.imread("D:/Code/ComputerVision/calibration_img/IMG_20210311_193203.jpg")
##extract corner point
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
#cv2.drawChessboardCorners(img, (w, h), corners, ret)
#height, width = img.shape[:2]  
#size = (int(width*0.5), int(height*0.5))  
#shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
#cv2.imshow("img1", img)
#cv2.imshow("img", shrink)
#cv2.waitKey(0)
#cv2.imwrite("./example1.jpg", img)


##Intrinsic Matrix
mtx = np.array([[3.28564699e+03, 0.00000000e+00, 1.98608659e+03], 
                [0.00000000e+00, 3.27653825e+03, 1.43982197e+03], 
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = "float32")
##Distortion Coefficient
dist = np.array([[-2.59113373e-02, 3.63905138e-01, -1.57290643e-04, 2.54486399e-04, -8.74978132e-01]], dtype = "float32")
img = cv2.imread("D:/Code/ComputerVision/calibration_img/IMG_20210311_193203.jpg")
h, w = img.shape[:2]
h1 = h 
w1 = w
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
print (newcameramtx)
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
#cv2.imwrite('./corrected_example1/example1_cali.jpg', dst1)
print ("dst:", dst1.shape)
resize_img = cv2.resize(dst1, (w1, h1))
cv2.imwrite("./corrected_example1/example1_cali_resize.jpg", resize_img)
print ("resize:", resize_img.shape)