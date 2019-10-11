import  platform
py_version = platform.python_version()
if py_version.startswith("3.5"):
    from .py35 import seetaface
elif py_version.startswith("3.6"):
    from .py36 import seetaface
elif py_version.startswith("3.7"):
    from .py37 import seetaface
elif py_version.startswith("2.7"):
    from .py27 import seetaface
else:
    print("[EOORR] The current python version:{} is not supported!".format(py_version))

import numpy  as np

class SeetaFaceEngine():
    def __init__(self,fd_path:str,fp_path:str, fr_path:str):
        """
        获取人脸识别引擎
        :param fd_path:
        :param fp_path:
        :param fr_path:
        """
        self.engine = seetaface.SeetaFace(fd_path,fp_path, fr_path)

    def detect_face(self,image:np.array) ->list:
        """
        人脸检测
        :param image: 原始大图
        :return:  人脸检测坐标[[x,y,w,h],...]
        """
        return  self.engine.detect_face(image)

    def detect_ponits(self,image:np.array,face_rect:list) ->list:
        """
        提取人脸关键点（5点）
        :param image: 原始大图
        :param face_rect: 人脸举行框 [ x,y,w,h]
        :return: 单个人脸的5个关键点位置 [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]
        """
        return  self.engine.detect_ponits(image,face_rect)

    def extract_feature(self,image:np.array,face_points:list)->np.array:
        """
        人脸特征提取，通过人脸关键点，函数内部将人脸做对应仿射变换达到人脸矫正目的，再对人脸提取特征值，增加人脸识别准确度
        :param image: 原始大图
        :param face_points: 人脸关键点坐标位置
        :return:
        """
        return  self.engine.extract_feature(image,face_points)

    def compare_feature(self,feature1:np.array,feature2:np.array) ->float:
        """
        人脸特征比较
        :param feature1: 人脸特征值1
        :param feature2: 人脸特征值2
        :return: 人脸相似度
        """
        return self.engine.compare_feature(feature1,feature2)

    def compare_feature_np(self,feature1:np.array,feature2:np.array) ->float:
        """
        使用numpy 计算，比较人脸特征值相似度
       :param feature1: 人脸特征值1
        :param feature2: 人脸特征值2
        :return: 人脸相似度
        """
        dot = np.sum(np.multiply(feature1, feature2))
        norm = np.linalg.norm(feature1) * np.linalg.norm(feature2)
        dist = dot / norm
        return float(dist)


    def compare_pair_images_demo(self,image1:np.array,image2:np.array) ->float:
        """
        两张图片快速比较样例，直接比较两张图片中人脸相似度,
        单张图片中没有检测到人脸或者检测人脸有多个则会抛出异常
        :param image1: 原始d大图像1
        :param image2: 原始大图像2
        :return: 人脸相似度
        """
        return self.engine.compare_pair_images_demo(image1,image2)

    def __str__(self):
        return "SeetaFaceEngine"