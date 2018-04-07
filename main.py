
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import facenet.src.align.detect_face

from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

#  import other libraries
import cv2

import tensorflow as tf
import time
import numpy as np

# Facenet parameters
gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

# Keras age & gender classification parameters
face_size = 64
age_gender_model = WideResNet(face_size, depth=16, k=8)()
age_gender_model_dir = os.path.join(os.getcwd(), 'pretrained_models').replace('//','\\')
image_width = 1280
image_height = 720

fpath = get_file('weights.18-4.06.hdf5', 'https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5',
                 cache_dir=age_gender_model_dir)

age_gender_model.load_weights(fpath)

# Address the error: ValueError: Tensor Tensor("dense_1/Softmax:0", shape=(?, 2), dtype=float32) is not an element of this graph.
age_gender_model._make_predict_function()

#   Start code from facenet/src/compare.py
print('Creating networks and loading parameters')
with tf.Graph().as_default():

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
        log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = facenet.src.align.detect_face.create_mtcnn(
            sess, None)

    cap = cv2.VideoCapture(0)
    cap.set(3, image_width)
    cap.set(4, image_height)

    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (image_width, image_height))

    frame_no = 0

    prev_time = time.time()

    resize_ratio = 0.5

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:

            frame_no += 1

            # Resize the frame for higher frame rate
            frame_resize = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio)

            height, width = frame_resize.shape[:2]

            start_time = time.time()

            # FaceNet return the bounding boxes and related landmarks
            bounding_boxes, points = facenet.src.align.detect_face.detect_face(
                frame_resize, minsize, pnet,
                rnet, onet, threshold, factor)

            end_time = time.time()

            # Draw the landmarks if any
            if points.any():

                for i in range(0, int(len(points)/2)):

                    if len(points[i]) == 1:

                        cv2.circle(frame, (points[i]/resize_ratio, points[i+5]/resize_ratio), 3, (0, 200, 0), -1)

            faces = np.empty((len(bounding_boxes), face_size, face_size, 3))

            # Iterate over each box
            index = 0
            for (x1, y1, x2, y2, acc) in bounding_boxes:

                y1_refined = int(y1/resize_ratio)
                y2_refined = int(y2/resize_ratio)
                x1_refined = int(x1/resize_ratio)
                x2_refined = int(x2/resize_ratio)

                cv2.rectangle(frame, (x1_refined, y1_refined), (x2_refined, y2_refined), (150, 150, 150), 1)

                if x1_refined >= 0 and y1_refined >=0 and x2_refined < image_width and y2_refined < image_height:

                    faces[index, :, :, :] = cv2.resize(frame[y1_refined:y2_refined+1, x1_refined:x2_refined+1, :], (face_size, face_size))

                index += 1

            # Apply age & gender classification every 3 frames
            if len(faces) is not 0 and frame_no % 3 == 0:

                results = age_gender_model.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                for i, (x1, y1, x2, y2, acc) in enumerate(bounding_boxes):

                    y1_refined = int(y1 / resize_ratio)
                    y2_refined = int(y2 / resize_ratio)
                    x1_refined = int(x1 / resize_ratio)
                    x2_refined = int(x2 / resize_ratio)

                    cv2.putText(img=frame, text='{}:{}'.format(int(predicted_ages[i]), "F" if predicted_genders[i][0] > 0.5 else "M"),
                            org=(x1_refined,y1_refined), fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA, fontScale=(y2-y1) * 0.02)

                    if predicted_genders[i][0] > 0.5:

                        cv2.rectangle(frame, (x1_refined, y1_refined),
                                  (x2_refined,
                                   y2_refined), (0, 0, 255), 2)
                    else:

                        cv2.rectangle(frame, (x1_refined, y1_refined),
                                      (x2_refined,
                                       y2_refined), (255, 0, 0), 2)

            current_time = time.time()

            # Write image resolution info
            cv2.putText(img=frame, text='Size: {} x {}'.format(height, width),
                        org=(30, 30), fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255), lineType=cv2.LINE_AA,
                        fontScale=1.7, thickness=2)

            # Write frame rate info
            cv2.putText(img=frame, text='FPS: {:.2f}'.format(1.0/(current_time - prev_time)),
                        org=(30, 70), fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255), lineType=cv2.LINE_AA,
                        fontScale=1.7, thickness=2)

            prev_time = current_time

            cv2.imshow('Frame', frame)

            writer.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):

                break

    cap.release()
    writer.release()
