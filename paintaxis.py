import numpy as np
#import cv2
import glob
from PIL import Image, ImageDraw, ImageFont
import cv2

images = glob.glob("./calibration_img/*.jpg")
fname = images[0]
img = Image.open(fname)
draw = ImageDraw.Draw(img)



####PAINT SQUARE
###TAKE THE FIRST IMAGE TO PAINT AXIS
u1 = 464.88055
v1 = 717.33386
u2 = 979.77985
v2 = 717.45460
u3 = 439.45230
v3 = 1197.5262
u4 = 971.88150
v4 = 1187.2756


###PAINT LINE
##line one 1-2
draw.line((u1, v1, u2, v2), width = 18, fill = 'cyan')
##line two 1-3
draw.line((u1, v1, u3, v3), width = 18, fill = 'cyan')
##line three 3-4
draw.line((u3, v3, u4, v4), width = 18, fill = 'cyan')
##line four 2-4
draw.line((u2, v2, u4, v4), width = 18, fill = 'cyan')

#img.show()


###COMPUTE LOCATION
##transform vector to rotation matrix
rvec = np.array([[-0.22911271], [-0.0960063 ], [-0.03824228]])
rmat = cv2.Rodrigues(rvec)
R = rmat[0]
#init
locationC = np.zeros((3, 1),dtype = 'float32')#camera
locationW = np.zeros((3, 1),dtype = 'float32')#world
locationP = np.zeros((3, 1),dtype = 'float32')#picture
#intrinsic matrix
K = np.array([[3.28564699e+03, 0.00000000e+00, 1.98608659e+03],
              [0.00000000e+00, 3.27653825e+03, 1.43982197e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]).reshape(3, 3)
#translation matrix
T = np.array([[-6.08063303], [-2.89240427], [13.17671622]]).reshape(3, 1)
##compute point5
locationW = np.array([0, 0, -2]).reshape(3, 1)
locationC = np.dot(R, locationW) + T
locationP = (1 / locationC[2][0]) * np.dot(np.dot(K,np.hstack((R,T))), np.vstack((locationW, 1)).reshape(4, 1))
u5 = locationP[0][0]
v5 = locationP[1][0]
##compute point6
locationW = np.array([2, 0, -2]).reshape(3, 1)
locationC = np.dot(R, locationW) + T
locationP = (1 / locationC[2][0]) * np.dot(np.dot(K,np.hstack((R,T))), np.vstack((locationW, 1)).reshape(4, 1))
u6 = locationP[0][0]
v6 = locationP[1][0]
##compute point7
locationW = np.array([0, 2, -2]).reshape(3, 1)
locationC = np.dot(R, locationW) + T
locationP = (1 / locationC[2][0]) * np.dot(np.dot(K,np.hstack((R,T))), np.vstack((locationW, 1)).reshape(4, 1))
u7 = locationP[0][0]
v7 = locationP[1][0]
##compute point8
locationW = np.array([2, 2, -2]).reshape(3, 1)
locationC = np.dot(R, locationW) + T
locationP = (1 / locationC[2][0]) * np.dot(np.dot(K,np.hstack((R,T))), np.vstack((locationW, 1)).reshape(4, 1))
u8 = locationP[0][0]
v8 = locationP[1][0]


###PAINT LINE
##line five 1-5
#draw.line((u1, v1, u5, v5), width = 18, fill = 'cyan')
##line six 2-6
#draw.line((u2, v2, u6, v6), width = 18, fill = 'cyan')
##line seven 3-7
#draw.line((u3, v3, u7, v7), width = 18, fill = 'cyan')
##line eight 4-8
#draw.line((u4, v4, u8, v8), width = 18, fill = 'cyan')
##line nine 5-6
#draw.line((u5, v5, u6, v6), width = 18, fill = 'cyan')
##line ten 5-7
#draw.line((u5, v5, u7, v7), width = 18, fill = 'cyan')
##line eleven 7-8
#draw.line((u7, v7, u8, v8), width = 18, fill = 'cyan')
##line twelve 6-8
#draw.line((u6, v6, u8, v8), width = 18, fill = 'cyan')

#img.show()
#img.save("./square.jpg")


####PAINT AXIS
###TAKE THE FIRST IMAGE TO PAINT AXIS
uo = 464.88055
vo = 717.33386
uy1 = u2
vy1 = v2
uy2 = 1231.3833
vy2 = 716.68005
ux1 = u3
vx1 = v3
ux2 = 421.7783
vx2 = 1446.2625
uz1 = u5
vz1 = v5
locationW = np.array([0, 0, -3]).reshape(3, 1)
locationC = np.dot(R, locationW) + T
locationP = (1 / locationC[2][0]) * np.dot(np.dot(K,np.hstack((R,T))), np.vstack((locationW, 1)).reshape(4, 1))
uz2 = locationP[0][0]
vz2 = locationP[1][0]
#draw.line((ux1, vx1, ux2, vx2), width = 20, fill = 'red')
font = ImageFont.truetype("consola.ttf", 88, encoding="unic")
#draw.text((ux2 + 2, vx2 + 2), u'X axis', 'red', font)
#draw.line((uy1, vy1, uy2, vy2), width = 20, fill = 'red')
#draw.text((uy2 + 2, vy2 + 2), u'Y axis', 'red', font)
#draw.line((uz1, vz1, uz2, vz2), width = 20, fill = 'red')
#draw.text((uz2 + 2, vz2 - 88), u'Z axis', 'red', font)

img.show()
img.save("./base.jpg")