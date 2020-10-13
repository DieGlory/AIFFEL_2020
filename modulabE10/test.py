# 하나는 너무 적소 5명으로 합시다! 땡큐 5명!
import matplotlib.pyplot as plt
import cv2
import os
import dlib
import numpy as np

detector_hog = dlib.get_frontal_face_detector()   #- detector 선언
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

my_image_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/test.jpg'
img_bgr = cv2.imread(my_image_path)    #- OpenCV로 이미지를 읽어서
img_show = img_bgr.copy()      #- 출력용 이미지 별도 보관
# plt.imshow 이전에 RGB 이미지로 바꾸는 것을 잊지마세요. 
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

dlib_rects = detector_hog(img_rgb, 1)

sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/king.png'
img_sticker = cv2.imread(sticker_path)
# img_sticker = cv2.resize(img_sticker, (w,h))
# print (img_sticker.shape)

for i in range(len(dlib_rects)):

    # print("dlib_rect : ",dlib_rect)
    
    list_landmarks = []
    print(dlib_rects[i])
    # for dlib_rect_elem in dlib_rects[i]:
    points = landmark_predictor(img_rgb, dlib_rects[i])
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

    x = list_landmarks[0][30][0]
    y = list_landmarks[0][30][1] - dlib_rects[i].width()//2
    w = dlib_rects[i].width()
    h = dlib_rects[i].width()
    
    temp_img_sticker = cv2.resize(img_sticker, (w,h))

    refined_x = x - w // 2  # left
    refined_y = y - h       # top

    temp_img_sticker = temp_img_sticker[-refined_y:]
    refined_y = 0

    sticker_area = img_show[refined_y:temp_img_sticker.shape[0], refined_x:refined_x+temp_img_sticker.shape[1]]
    img_show[refined_y:temp_img_sticker.shape[0], refined_x:refined_x+temp_img_sticker.shape[1]] = np.where(temp_img_sticker==0,sticker_area,temp_img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()