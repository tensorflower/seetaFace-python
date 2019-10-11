import  SeetaFacePy
import cv2
engine = SeetaFacePy.SeetaFaceEngine("models/fd_2_00.dat","models/pd_2_00_pts5.dat","models/fr_2_10.dat")

image1 = cv2.imread("images/1.jpg")
image2 = cv2.imread("images/2.jpg")

face_rect_list1 = engine.detect_face(image1)
face_rect_list2 = engine.detect_face(image2)

face_ponits1 = engine.detect_ponits(image1,face_rect_list1[0])
face_ponits2 = engine.detect_ponits(image2,face_rect_list2[0])

feature1 = engine.extract_feature(image1,face_ponits1)

feature2 = engine.extract_feature(image2,face_ponits2)

similar =  engine.compare_feature(feature1,feature2)

np_similar =  engine.compare_feature_np(feature1,feature2)

demo_similar = engine.compare_pair_images_demo(image1,image2)

#print  result

print("face_rect_list1:",face_rect_list1)
print("face_rect_list2:",face_rect_list2)
print("face_ponits1:",face_ponits1)
print("face_ponits2:",face_ponits2)
print("similar:",similar)
print("np_similar:",similar)
print("demo_similar:",demo_similar)

#draw
cv2.rectangle(image1,(face_rect_list1[0][0],face_rect_list1[0][1]),(face_rect_list1[0][0]+face_rect_list1[0][2],face_rect_list1[0][1]+face_rect_list1[0][3]),(255,0,0),1)
cv2.rectangle(image2,(face_rect_list2[0][0],face_rect_list2[0][1]),(face_rect_list2[0][0]+face_rect_list2[0][2],face_rect_list2[0][1]+face_rect_list2[0][3]),(255,0,0),1)

for  index,point in  enumerate(face_ponits1):
    cv2.circle(image1,(int(point[0]),int(point[1])),2,(255,255,255),-1)
    cv2.putText(image1,str(index+1),(int(point[0]),int(point[1])),1,1,(0,0,255))

for  index,point in  enumerate(face_ponits2):
    cv2.circle(image2,(int(point[0]),int(point[1])),2,(255,255,255),-1)
    cv2.putText(image2,str(index+1),(int(point[0]),int(point[1])),1,1,(0,0,255))


cv2.imshow("image1",image1)
cv2.imshow("image2",image2)
cv2.waitKey(0)