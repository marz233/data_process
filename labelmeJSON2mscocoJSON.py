#-*- coding:utf-8 -*-
# !/usr/bin/env python
import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
# label_name_to_value = {'_background_': 0,"abrasion": 1,"burn": 2,
#                                 "dent": 3,"missing_tbc": 4,"crack": 5,
#                                 "curl": 6,"nick": 7,"drop_piece": 8,
#                                 "hole": 9,"missing_material": 10}##########

label_name_to_value = {'_background_': 0,'abrasion': 1, 'burn': 2,
                                 'crack': 3, 'curl': 4, 'dent': 5,
                                 'drop_piece': 6, 'hole': 7, 'missing_material': 8,
                                 'missing_tbc': 9, 'nick': 10}###
class labelme2coco(object):
    
    def __init__(self,labelme_json=[],save_json_path='./new.json'):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json=labelme_json
        self.save_json_path=save_json_path
        self.images=[]
        self.categories=[]
        self.annotations=[]
        # self.data_coco = {}
        self.label=[]
        self.annID=1
        self.height=0
        self.width=0

        self.save_json()

    def data_transfer(self):
        for num,json_file in enumerate(sorted(self.labelme_json)):
            with open(json_file,'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data,num))
                for shapes in data['shapes']:
                    label=shapes['label']##.split('_')
                    if label not in self.label:##
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    
#                     for i in range(len(shapes['points'])):
#                         print(type(shapes['points'][i]))
                    
                    temp = np.array(shapes['points'])#list转numpy.array
                    temp=np.round(temp).astype(float)#四舍五入然后转为整数
                    arr = temp.tolist()#numpy.array转list
#                     print(type(arr))
                    points=arr
                    
                    self.annotations.append(self.annotation(points,label,num))
                    self.annID+=1

    def image(self,data,num):
        image={}
#         img = utils.img_b64_to_arr(data['imageData'])  # 解析原图片数据
#         # img=io.imread(data['imagePath']) # 通过图片路径打开图片
#         # img = cv2.imread(data['imagePath'], 0)
#         height, width = img.shape[:2]
        height=data["imageHeight"]
        width=data["imageWidth"]
        img = None
        image['height']=height
        image['width'] = width
        image['id']=num+1
        image['file_name'] = data['imagePath'].split('\\')[-1]##split取了‘\\’分隔的data['imagePath']中倒数第一个部分
#         image['file_name']=
        self.height=height
        self.width=width

        return image

    def categorie(self,label):
        categorie={}
        categorie['supercategory'] = label
#         categorie['id']=len(self.label)+1 # 0 默认为背景
        categorie['id']=label_name_to_value[label]########
        categorie['name'] = label##
        return categorie
    
    def compute_polygon_area(self,points):##def compute_polygon_area(points):参数列表少了self
        point_num = len(points)
        if(point_num < 3): return 0.0
        s = points[0][1] * (points[point_num-1][0] - points[1][0])
        for i in range(1, point_num):
            s += points[i][1] * (points[i-1][0] - points[(i+1)%point_num][0])
        return abs(s/2.0)
    
    def annotation(self,points,label,num):
        annotation={}
        annotation['segmentation']=[list(np.asarray(points).flatten())]
    #     poly = Polygon(points)
    #     area_ = round(poly.area,6)
#         print(points);type(points);
        annotation['area'] = self.compute_polygon_area(points)
        annotation['iscrowd'] = 0
        annotation['image_id'] = num+1

        annotation['bbox'] = list(map(float,self.getbbox(points)))

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self,label):
        
        for categorie in self.categories:
            if label==categorie['name']:##
                return label_name_to_value[label]
#         return categorie['id']

        return -1

    def getbbox(self,points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height,self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c-left_top_c, right_bottom_r-left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self,img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco={}
        data_coco['images']=self.images
        data_coco['categories']=self.categories
                
        data_coco['annotations']=self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w', encoding="utf-8"), indent=4,ensure_ascii=False)#ensure_ascii=False)  # indent=4 更加美观显示
##添加ensure_ascii=False显示中文
# labelme_json=glob.glob('./*.json')
# labelme2coco(labelme_json,'./new.json')
labelme_json=glob.glob(r'/home/cver/lcx/data/augmentation/guass_json/*.json')
labelme2coco(labelme_json,r'/home/cver/docker-data/maskrcnn-lcx/datasets/annotations/instances_train2017.json')
