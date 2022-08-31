# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 00:39:15 2022

@author: Kang, Yong-sik

관로를 점검한 파일 또는 동영상을 입력 받아 파손 유무 및 파손 유형을 판단
동영상은 프레임을 계산하고 1초 단위로 이미지를 추출 후 이미지를 모델에 입력하여 
파손 유형을 판단
"""

import streamlit as st
import os
import cv2
from PIL import Image
import math
import numpy as np
import tensorflow as tf

from tensorflow import keras

def save_uploadedfile (uploaded_file) :
    with open (os.path.join(temp_dir, uploaded_file.name), "wb") as f :
        f.write (uploaded_file.getbuffer())
    return st.success("Saved File:{} to Server".format(uploaded_file.name))        

def extract_img_from_file (uploaded_file) :
    file_path = temp_dir + uploaded_file.name
    st.write (file_path)
    video = cv2.VideoCapture(file_path)

    if not video.isOpened() :
        print ('Could not open :', file_path)
        exit(0)
    
    length = int (video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int (video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int (video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS)) 

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Length", length)
    col2.metric("Width", width)
    col3.metric("Height", height)
    col4.metric("FPS", fps)

#    st.write ('length : ', length)
#    st.write ('width : ', width)
#    st.write ('height : ', height)
#    st.write ('fps : ', fps)

    try : 
        if not os.path.exists(file_path[:-4]) : 
            os.makedirs (file_path[:-4])
    except OSError :
                st.write ('Error : Creating directory. ' + file_path[:-4])
    
    count = 0

    ret = True

    while (video.isOpened() and ret == True) :
        ret, image = video.read ()
        
        if (int(video.get(1)) % fps == 0) :
            cv2.imwrite(file_path[:-4]+ '/frame%d.jpg' % count, image)
#            st.write ('Saved frame number : ', str(int(video.get(1))))
            count +=1
#            st.write (ret) 
        
    video.release()
#    st.write ('video was released')
    
    return count

def pipe_check (img_path) :
    
    img = keras.preprocessing.image.load_img (img_path, target_size = (img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array (img)
    img_array = tf.expand_dims(img_array,0)

    predictions = reconstructed_model.predict (img_array)
    score = tf.nn.softmax (predictions[0])

#    print ('This image most likely belongs to {} with a {:.2f} percent confidence.'.format (class_name[np.argmax(score)], 100*np.max(score)))    
#    msg = 'This image most likely belongs to {} with a {:.2f} percent confidence.'.format (class_name[np.argmax(score)], 100*np.max(score))
    msg = '파손유형 : {} \n예측 정확도 : {:.2f}'.format (class_name[np.argmax(score)], 100*np.max(score))
    st.write ( 'ㅁ 파손유형 : {}'.format (class_name[np.argmax(score)]))
    st.write ( 'ㅁ 예측 정확도 : {:.2f}'.format (100*np.max(score)))
    
    return msg


# 메인 시작

st.title ("관로 파손 유형 점검")
# st.header ("관로 파손 점검 웹")
st.write ("관로 내부를 촬영한 이미지 또는 동영상을 입력 받아 파손 유형을 판단 합니다.")


# pipe_check 학습 모델 로드


img_height = 224
img_width = 224
batch_size = 32

data_dir = 'C:/Users/kwater/python/pipe_data/Training2'
temp_dir ='C:/Users/kwater/temp/'

class_name =  os.listdir (data_dir)

reconstructed_model = keras.models.load_model ('pipe_class')

select = st.radio ('이미지 분석 또는 동영상 분석을 선택하세요', ('이미지', '동영상'))

if select == '이미지' :
    st.write ('이미지 분석을 선택하셨습니다.')
    uploaded_file = st.file_uploader ("이미지 파일을 선택해 주세요", type = ["jpg", "png", "gif"])    
    if uploaded_file is not None :
        file_details = {"FileName": uploaded_file.name, "File Type" : uploaded_file.type }
        st.write(file_details)    
        save_uploadedfile(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1 :
#         st.write (temp_dir + uploaded_file.name)
         img_path = temp_dir + uploaded_file.name
         image = Image.open (img_path)
         st.image (image)         
         msg = pipe_check (img_path)
#         st.write (msg)



else :
    st.write ('동영상 분석을 선택하셨습니다.')
    uploaded_file = st.file_uploader ("동영상 파일을 선택해 주세요", type = ["mp4", "mpg", "wmv", "mov"])        
    if uploaded_file is not None :
        file_details = {"FileName":uploaded_file.name, "File Type" : uploaded_file.type }
        st.write(file_details)
        save_uploadedfile(uploaded_file)    
        count = extract_img_from_file (uploaded_file)

        img_dir = temp_dir + uploaded_file.name[:-4] + '/'
#        st.write (img_dir)
        imgno = 0
        
       
        for i in range (0, math.ceil(count/4)) :

            col1, col2, col3, col4 = st.columns(4)
            with col1 :
                img_name = 'frame' + str(imgno) + '.jpg'
                img_path = img_dir + 'frame' + str(imgno) + '.jpg'
                st.write (img_name)
                image = Image.open (img_path)
                st.image (image)            
                msg = pipe_check (img_path)
#                st.write (msg)                
               
                imgno += 1
                if imgno == count :
                    break

            with col2 :
                img_name = 'frame' + str(imgno) + '.jpg'
                img_path = img_dir + 'frame' + str(imgno) + '.jpg'
                st.write (img_name)
                image = Image.open (img_path)
                st.image (image)            
                msg = pipe_check (img_path)
 #               st.write (msg)                

                imgno += 1                
                if imgno == count :
                    break
                
                
            with col3 :
                img_name = 'frame' + str(imgno) + '.jpg'
                img_path = img_dir + 'frame' + str(imgno) + '.jpg'
                st.write (img_name)                
                image = Image.open (img_path)
                st.image (image)            
                msg = pipe_check (img_path)
#                st.write (msg)                

                imgno += 1
                if imgno == count :
                    break

            with col4 :
                img_name = 'frame' + str(imgno) + '.jpg'
                img_path = img_dir + 'frame' + str(imgno) + '.jpg'
                st.write (img_name)                
                image = Image.open (img_path)
                st.image (image)            
                msg = pipe_check (img_path)
#                st.write (msg)                

                imgno += 1
                if imgno == count :
                    break
        
