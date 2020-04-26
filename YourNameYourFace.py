# 你的名字，你的face
'''
## PaddleHub：人脸检测主题创意赛。实现自动检测人脸，然后使用《你的名字》动漫中的图片拼出人脸。
#### 本次主要使用 **PaddleHub** 开源的人脸关键点检测模型 
    [face_landmark_localization](https://www.paddlepaddle.org.cn/hubdetail?name=face_landmark_localization&en_category=KeyPointDetection) 进行人体关键点识别。
#### 关键点识别参考案例：
    [PaddleHub实战——人像美颜](https://aistudio.baidu.com/aistudio/projectdetail/389512)。
#### 之后再以 RGB 颜色均值为标准进行人脸拼图。
#### **NOTE**： 本项目在百度AIStudio实现，如果需要在本地运行该项目示例，
                首先要安装PaddleHub。
                其中 face_landmark_localization 使用1.0.2版， 
                paddlepaddle 环境为1.6.2， 
                paddlehub 版本为1.6.1
'''

## 一、加载图片，检测关键点
#### 检测关键点，并将关键点以红色点状的方式画在原图上，保存并显示。
import cv2
import paddlehub as hub
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import math

src_img = cv2.imread('single_face.jpg')

module = hub.Module(name="face_landmark_localization")
result = module.keypoint_detection(images=[src_img])

tmp_img = src_img.copy()
for index, point in enumerate(result[0]['data'][0]):
    # cv2.putText(img, str(index), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), -1)
    cv2.circle(tmp_img, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

res_img_path = 'face_landmark.jpg'
cv2.imwrite(res_img_path, tmp_img)

img = mpimg.imread(res_img_path) 
# 展示预测68个关键点结果
plt.figure(figsize=(8,8))
plt.imshow(img) 
plt.axis('off') 
plt.show()

## 二、取出目标区域
#### 对关键点坐标值做一个大小判断，取出待重组的图像区域。

'''
# 解压图片
import zipfile

path_zip = "./data/data31722/your_name.zip" # 所需图片的存放路径
z = zipfile.ZipFile(path_zip, "r")   # 读取zip文件
num_Image = len(z.namelist()) - 1    # 总的图片数量

# 数据解压
path_Image = './data/'
#with zipfile.ZipFile(path_zip, 'r') as zin:
#    zin.extractall(path_Image)
'''
num_Image = 638    # 总的图片数量
points = np.mat(result[0]['data'][0])
# 获取待重组的矩形局域的两个坐标值
point_a = np.floor(np.amin(points, axis=0))
point_d = np.ceil(np.amax(points, axis=0))

# 计算目标区域所占大小，以及用与重组的图像应该缩放为多大
ROI_area = (point_d[0, 0] - point_a[0, 0])*(point_d[0, 1] - point_a[0, 1])
# 每张图片不重复，则需要的每张图片所占的大小，再除以10
single_image_area = (ROI_area / num_Image) / 10

## 三、组合图片
#### 计算特征向量，以 R/G/B 三个通道分别求平均值作为特征。

path_Images =  'F:/picture/your_name/out'

# 压缩图像至固定大小
def pic_compression(src_pic):
    target_high = src_pic.shape[0]
    target_weight = src_pic.shape[1]
    if target_high < target_weight:
        target_high = np.floor(np.sqrt(single_image_area /  (target_weight/target_high)))
        target_weight = np.ceil(single_image_area / target_high)
    else:
        target_weight = np.floor(np.sqrt(single_image_area /  (target_high/target_weight)))
        target_high = np.ceil(single_image_area / target_weight)
    return cv2.resize(src_pic,(int(target_weight),int(target_high))), target_high, target_weight

# 计算图像的RGB特征
feature_dim = 3 # 特征维度 R、G、B 共三维
pic_feature = np.zeros([num_Image,feature_dim]) # 特征向量
for indexImg in range(3,num_Image+3-1):  # 图片索引由3开始
    path_pic = path_Images + str(indexImg) +'.png' # 获取每张图片的地址
    pic = cv2.imread(path_pic)
    pic_comed,th,tw = pic_compression(pic.copy()) # 计算得到压缩图像
    for idx in range(len(pic_feature[0])):
        pic_feature[indexImg-3,idx] = np.average(pic_comed[:-1,:-1,idx])

#### 计算脸部各块的RGB特征并贴图
# 抠出待组合区域图像
left_p = int(point_a[0, 0])
right_p = int(point_d[0, 0])
top_p = int(point_a[0, 1])
bottom_p = int(point_d[0, 1])
# 得到目标图像
temp_Image = src_img.copy()
ROI_Image = temp_Image[top_p:bottom_p, left_p:right_p]

block_feature = np.zeros(feature_dim) # 每一图块的特征
blo_fea_buff = np.zeros(num_Image)  # 缓存每一图块特征到所有图片特征的欧式距离值
for idx_i in range(0, len(ROI_Image)-int(th), int(th)):
    for idx_j in range(0, len(ROI_Image[1])-int(tw), int(tw)):
        for idx in range(len(pic_feature[0])):
            block_feature[idx] = np.average(ROI_Image[idx_i:idx_i+int(th),idx_j:idx_j+int(tw),idx])
        for img_idx in range(0,num_Image):
            blo_fea_buff[img_idx] = np.linalg.norm(block_feature - pic_feature[img_idx])
        pic_idx = np.argmin(blo_fea_buff) + 3 # 获取到最小欧式距离的图片索引
        path_pic = path_Images + str(int(pic_idx)) +'.png'
        pic = cv2.imread(path_pic)
        pic_comed,_,_ = pic_compression(pic.copy()) # 计算得到压缩图像
        ROI_Image[idx_i:idx_i+int(th),idx_j:idx_j+int(tw)] = pic_comed

##### 将重组得到的脸按形状贴到原图上

def mask(image, face_landmark):
    """
    image： 人像图片
    face_landmark: 人脸关键点
    """
    image_cp = image.copy()
    hull = cv2.convexHull(face_landmark)

    cv2.fillPoly(image, [hull], (0, 0, 0))
    for idx_i in range(top_p,bottom_p):
        for idx_j in range(left_p,right_p):
            if (image[idx_i, idx_j] == [0,0,0]).all():
                image[idx_i, idx_j] = ROI_Image[idx_i - top_p,idx_j - left_p]
    #cv2.drawContours(image, [hull], -1, ROI_Image, -1)
    #cv2.addWeighted(image, 0.2, image_cp, 0.9, 0, image_cp)

    return image

# 获取人脸关键点数据，和原始图像
face_landmark = np.array(result[0]['data'][0], dtype='int')
result_image = mask(src_img.copy(), face_landmark)

cv2.imwrite('result.jpg', result_image)

img = mpimg.imread('result.jpg') 
plt.figure(figsize=(10,10))
plt.imshow(img) 
plt.axis('off') 
plt.show()
