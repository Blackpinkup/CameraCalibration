import cv2
import numpy as np
import glob
from PIL import Image, ImageDraw

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((13 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:13, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("./img/*.jpg")
i = 0
for fname in images:
    img = cv2.imread(fname)
    i += 1
    #cv2.imshow('img',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (13, 6), None)
    if ret:
        print(str(i) + " True")

    if ret:

        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

#cv2.drawChessboardCorners(img, (13, 6), corners, ret)# 记住，OpenCV的绘制函数一般无返回值
#cv2.imshow('img', img)
#cv2.waitKey(10)
print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
#rmats = np.zeros((3,3,32))
#for i in range(32):
    #cv2.Rodrigues(rvecs[i], rmats[i])

print("ret:", ret)
#print("mtx:\n", mtx) # 内参数矩阵
#print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
#print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
#print("rmats:\n", rmats)
#print("tvecs:\n", tvecs ) # 平移向量  # 外参数
#print("img1_locations(2D):\n")
#print(img_points[0])
#print("\n")
#print("world1_locations(3D):\n")
#print(obj_points[0])
#print("\n")

#print("-----------------------------------------------------")
img1 = cv2.imread("D:\Code\ComputerVision\AssignmentThree\PSMNet-master\dataset\my_4\IMG_20210420_211733.jpg")
h, w = img1.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
#print (newcameramtx)
dst = cv2.undistort(img1,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('D:\Code\ComputerVision\AssignmentThree\PSMNet-master\dataset\my_4\im0.jpg', dst1)

img2 = cv2.imread("D:\Code\ComputerVision\AssignmentThree\PSMNet-master\dataset\my_4\IMG_20210420_211735.jpg")
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
#print (newcameramtx)
dst = cv2.undistort(img2,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('D:\Code\ComputerVision\AssignmentThree\PSMNet-master\dataset\my_4\im1.jpg', dst1)
#print ("dst的大小为:", dst1.shape)