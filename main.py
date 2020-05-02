import cv2  as cv
import numpy as np
import math
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import os.path
from skimage import io,data
from skimage.measure import label,regionprops
#r"C:\Users\Vivien\Desktop\8.JPG"
image =  cv.imread(r"D:\grocery\dexian\paycheck\002s.JPG",1)
img = image.copy()
height_img, width_img, numbers = img.shape

# 有旋转的
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
img_median = cv.medianBlur(gray,5)
edges = cv.Canny(img_median, 50, 150, apertureSize=3)
lines = cv.HoughLines(edges,1,np.pi/180,0)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    if x1 != x2 or y1 != y2: 
        t = float(y2-y1)/(x2-x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        rotate_img = ndimage.rotate(img, rotate_angle)
binary,contours, hierarchy = cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE) 
for i in range(len(contours)):
    rect = cv.minAreaRect(contours[i])
    box = cv.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 左下角 左上角 右上角 右下角
    box = np.int0(box)
    width=np.sqrt(np.sum(np.square([box][0][0]-[box][0][3])))
    height=np.sqrt(np.sum(np.square([box][0][1]-[box][0][0])))
    if(height!=0 and width!=0):#参数是要根据情况调整的
        ratio = width / height
       
        if (ratio>1.5 and ratio < 2.5): 
            
#            print("width:")
#           print(width)
#            print(width_img)
#            print("height:")
#            print(height)
#            print(height_img)           
            if (width>0.2 * width_img and height > 0.1 * height_img): 
                cv.drawContours(img,[box],0,(0,0,255),2)
    else:
        continue
plt.imshow(rotate_img) # 可以用misc存倾斜调整后的照片

#没有旋转的
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
img_median = cv.medianBlur(gray,5) # 参数选取非常受限
edges = cv.Canny(img_median, 50, 150, apertureSize=3)
binary,contours, hierarchy = cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE) 
for i in range(len(contours)):
    x, y, width, height = cv.boundingRect(contours[i])
    ratio = width/height
    if (ratio>1.5 and ratio<3.5):        
        if(width>0.5*width_img and height > 0.3* height_img):
            rect = cv.minAreaRect(contours[i])
            box = cv.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 左下角 左上角 右上角 右下角
            box = np.int0(box)
            cv.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 2)
            contours_target = contours[i] #目标支票框
#cv.imshow("edges",edges)
#cv.imshow("blur",img_median)
cv.imshow("img",img)

#金额框
Xs = [i[0] for i in box] #BOX 装了目标框的四个顶点
Ys = [i[1] for i in box]
x1 = min(Xs)
y1 = min(Ys)
x2 = max(Xs)
y2 = max(Ys)
height_roi = y2 -y1
width_roi = x2 - x1
roi1 = edges[y1+2:y1+height_roi,x1+2:x1+width_roi] #用于后面
roi2 = image[y1+2:y1+height_roi,x1+2:x1+width_roi] #切出来的支票原彩图
h_roi2, w_roi2,num = img.shape

#cv.waitKey(0)
#cv.destroyAllWindows()

#直接根据比例框区域
lpt_x = (int)((w_roi2) *0.13)-5
lpt_y =(int)((h_roi2) * 0.22)-3
w=(int)((w_roi2) * 0.665)
h=(int)((h_roi2) * 0.10)
money = roi2[lpt_y:lpt_y+h,lpt_x:lpt_x+w]
cv.imshow("origin",money)

#图像亮度调整
#brightness average
std_y = 187.2
for i in range(h):
    for j in range(w):
        R=money[i,j][2]
        G=money[i,j][1]
        B=money[i,j][0]
        y = R*0.299 + G*0.587+B*0.114 + num
print(y)
if y < 0.8* std_y or y > 1.2*std_y:
    gap = std_y -y
    add= int(gap/0.98)
    for i in range(h):
        for j in range(w):
            R=money[i,j][2]
            G=money[i,j][1]
            B=money[i,j][0]
            # R
            if R+add >235:
                money[i,j][2]=255
            elif R+add<20:
                money[i,j][2]= 0
            else:
                money[i,j][2]= money[i,j][2]+add
            # G     
            if G+add >235:
                money[i,j][1]=255
            elif G+add<20:
                money[i,j][1]= 0
            else:
                 money[i,j][1]=  money[i,j][1]+add
            # B
            if B+add >235:
                money[i,j][0]=255
            elif B+add<20:
                money[i,j][0]= 0
            else:
                 money[i,j][0]=  money[i,j][0]+add
                
gray_money = cv.cvtColor(money,cv.COLOR_BGR2GRAY)
img_median_money = cv.medianBlur(gray_money,3)
ret, thresh = cv.threshold(img_median_money, 100, 255, 0) #阈值important 100
element = cv.getStructuringElement(cv.MORPH_CROSS,(2,2))
erosion = cv.erode(thresh,element,iterations =2) #used for searching for pixels
#反色
erosion_opp = erosion.copy()
cv.bitwise_not(erosion,erosion_opp)

show = money.copy()
#cv.imshow("roi2",roi2)
cv.imshow("money",money)
#cv.imshow("blur",img_median)
cv.imshow("erosion",erosion)
cv.imshow("or_thresh",thresh)
#cv.imshow("blur",img_median_money)

#字体切分，利用像素点
h_ero, w_ero = erosion.shape
black= []
white= []
#count pixel by column
for i in range(w_ero):
    col_white=0
    col_black=0
    for j in range(h_ero): # x,y 是先x后y，但是矩阵表示里，是先y后x，行排列
        if erosion[j][i] == 255: #white
            col_white+=1
        if erosion[j][i] == 0:
            col_black+=1
    black.append(col_black)
    white.append(col_white)
black_max = max(black)
white_max = max(white)
#configure 白底黑字
if black_max > white_max:
    config = False
else:
    config = True
binary_eroopp,contours_eroopp, hierarchy_eroopp = cv.findContours(erosion,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
target_ctr = []
temp = []
order= []
show1 = show.copy()
for i in range(len(contours_eroopp)):
    x, y, width, height = cv.boundingRect(contours_eroopp[i])
    ratio = width/height
    area  = cv.contourArea(contours_eroopp[i])
    if x<0.6 * w_ero and ratio <4: 
#        target_ctr.append(contours_eroopp[i])
        xc =int(0.5*width +x)
        yc=int(0.5*height + y)
        a=np.array([x,y,width,height,i])
        temp.append(a)
        #cv.rectangle(show,(x,y),(x+width,y+height),(255,0,0),1) #蓝色

#轮廓的顺序是乱的，影响后面连通域的融合
temp=np.asarray(temp)
#print("temp",temp)
order = temp[np.lexsort(temp[:,::-1].T)] #竖向切分，从小到大
#print("order",order)
#每一个连通域都去跟除自己之外的连通域比较
merge = []
print(order)
#print(order)
for i in range(len(order)):
    count = 0
    index = order[i][4] #contour里的顺序
    properities = [] #[0]表示是order里的顺序
    xi,yi,widthi,heighti = order[i][0],order[i][1],order[i][2],order[i][3]
#    print(i,"i",xi,yi,widthi,heighti)
    #从左向右分析，剔除重复比较,结束特征是0
    for j in range(i,len(order)):
        xj,yj,widthj,heightj = order[j][0],order[j][1],order[j][2],order[j][3]
        min_w = min(widthi,widthj)
#        print(j,"j",xj,yj,widthj,heightj)
        if j==i:
            continue
        if xj< xi+widthi and (yj+0.67*heighti < yi or yj > yi +0.67*heighti):
            if yj>yi and yj+heightj < yi+heighti:
                print("in")
                continue
            properities.append(j)
            count=1
            continue
        if xi+widthi-xj > 0.05*min_w:
            properities.append(j)
            count=1
            continue
        if xj-xi < min_w:
            properities.append(j)
            count=1
            print(count,i,j)
            continue
            
    if count == 0:
        properities.append(0)
    properities = np.asarray(properities)
    merge.append(properities)      
    cv.rectangle(show1,(xi,yi),(xi+widthi,yi+heighti),(255,0,0,),1)
    cv.imshow("show1",show1)
#    cv.waitKey(0)
print(merge) 
x=[]
y=[]
gap = []
flag=0
for i in range(len(merge)):
    if len(merge[i])>1 or merge[i]!=0:
        x.append(order[i][0])
        x.append(order[i][0]+order[i][2])
        y.append(order[i][1])
        y.append(order[i][1]+order[i][3])
        flag=1
    if len(merge[i]==1):
        if merge[i][0]==0 and flag ==1:
            x.append(order[i][0])
            x.append(order[i][0]+order[i][2])
            y.append(order[i][1])
            y.append(order[i][1]+order[i][3])
            x1,x2,y1,y2= min(x),max(x),min(y),max(y)
            gap.append(x2-x1)
            cv.rectangle(show,(x1,y1),(x2,y2),(255,0,0),1)
            x,y=[],[]
            cv.imshow("show",show)
            cv.waitKey(0)
            flag=0
            continue
        if merge[i][0]==0 and flag ==0:
            x1,x2,y1,y2= order[i][0],order[i][0]+order[i][2],order[i][1],order[i][1]+order[i][3]
            if x2-x1<0.03*w_ero or y2-y1<0.2*h_ero :
            #if x2-x1 <0.5*min(gap) or x2-x1>2*min(gap):
                continue
            cv.rectangle(show,(x1,y1),(x2,y2),(255,255,0),1)
            cv.imshow("show",show)
            cv.waitKey(0)
            
cv.imshow("show",show)
cv.waitKey(0)
cv.destroyAllWindows()
