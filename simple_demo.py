import cv2
import  SeetaFacePy
image1 = cv2.imread("images/1.jpg")
image2 = cv2.imread("images/2.jpg")
engine = SeetaFacePy.SeetaFaceEngine("models/fd_2_00.dat","models/pd_2_00_pts5.dat","models/fr_2_10.dat")
similar = engine.compare_pair_images_demo(image1,image2)
print("similar:",similar)