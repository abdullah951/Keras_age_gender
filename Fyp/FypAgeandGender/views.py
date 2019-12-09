from django.shortcuts import render
from contextlib import contextmanager

from django.shortcuts import render

# Create your views here.
import base64
from pathlib import Path

import os
import numpy as np
from time import sleep
from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from .wide_resnet import WideResNet
from .serializers import FileSerializer
import json

import cv2 as cv
import cv2
import math
import time
import argparse

from keras.utils.data_utils import get_file
from keras import backend as K

# Create your views here.

class FileView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FileView, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        K.clear_session()
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def post(self, request, *args, **kwargs):
        global image_data
        global encoded_string
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            # print(file_serializer.data.get('remark'))
            # info = json.loads(json.loads(file_serializer.data))
            # result = json.loads(str(file_serializer.data))
            print(file_serializer.data.get('file'))
            print(file_serializer.data.get('remark'))
            if file_serializer.data.get('remark') == 'image':

                face_cascade = cv2.CascadeClassifier(self.CASE_PATH)
                encoded_string=""
                facedetected = False
                # 0 means the default video capture device in OS
                video_capture = cv2.VideoCapture(r'F:/Keras_age_gender/Fyp' + file_serializer.data.get('file'))
                # infinite loop, break by key ESC
                while cv.waitKey(1) < 0:
                    if not video_capture.isOpened():
                        sleep(5)
                    # Capture frame-by-frame

                    ret, frame = video_capture.read()
                    if not ret:
                        cv.waitKey()
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=10,
                        minSize=(64, 64)
                    )
                    # placeholder for cropped faces
                    print(len(faces))
                    face_imgs = np.empty((len(faces), 64, 64, 3))
                    for i, face in enumerate(faces):
                        facedetected = True
                        face_img, cropped = self.crop_face(frame, face, margin=40, size=64)
                        (x, y, w, h) = cropped
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                        face_imgs[i, :, :, :] = face_img
                    if len(face_imgs) > 0:
                        # predict ages and genders of the detected faces
                        results = self.model.predict(face_imgs)
                        predicted_genders = results[0]
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()
                    # draw results
                    for i, face in enumerate(faces):
                        label = "{}, {}".format(int(predicted_ages[i]),
                                                "Female" if predicted_genders[i][0] > 0.5 else "Male")
                        self.draw_label(frame, (face[0], face[1]), label)

                    # cv2.imshow('Keras Faces', frame)
                    cv2.imwrite("output.png", frame)
                    if facedetected:

                        with open("output.png", "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read())
                            n=12

                            #m = {'type': 'image', 'base64': [encoded_string[i:i + n] for i in range(0, len(encoded_string), n)]}
                            m = {'type': 'image',
                                 'base64': encoded_string}
                            #print(len(m.get('base64')))
                    else:
                        encoded_string = None
                        m = {'type': 'image',
                             'base64': 'None'}
                        print(m.get('base64'))
                K.clear_session()
                return Response(encoded_string)

            elif file_serializer.data.get('remark') == 'video':

                face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

                # 0 means the default video capture device in OS
                video_capture = cv2.VideoCapture(r'F:/Keras_age_gender/Fyp' + file_serializer.data.get('file'))

                frame_width = int(video_capture.get(3))
                frame_height = int(video_capture.get(4))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter("output.avi", fourcc, 20.0,
                                      (frame_width, frame_height))
                encoded_string = ""
                facedetected = False
                # infinite loop, break by key ESC
                while cv.waitKey(1) < 0:
                    if not video_capture.isOpened():
                        sleep(5)
                    # Capture frame-by-frame

                    ret, frame = video_capture.read()
                    if not ret:
                        cv.waitKey()
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=10,
                        minSize=(64, 64)
                    )
                    # placeholder for cropped faces
                    face_imgs = np.empty((len(faces), 64, 64, 3))
                    for i, face in enumerate(faces):
                        facedetected = True;
                        face_img, cropped = self.crop_face(frame, face, margin=40, size=64)
                        (x, y, w, h) = cropped
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                        face_imgs[i, :, :, :] = face_img
                    if len(face_imgs) > 0:
                        # predict ages and genders of the detected faces
                        results = self.model.predict(face_imgs)
                        predicted_genders = results[0]
                        ages = np.arange(0, 101).reshape(101, 1)
                        predicted_ages = results[1].dot(ages).flatten()
                    # draw results
                    for i, face in enumerate(faces):
                        label = "{}, {}".format(int(predicted_ages[i]),
                                                "Female" if predicted_genders[i][0] > 0.5 else "Male")
                        self.draw_label(frame, (face[0], face[1]), label)

                    # cv2.imshow('Keras Faces', frame)
                    out.write(frame)
                    cv2.imwrite("output.png", frame)
                if facedetected:

                    with open("output.avi", "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read())
                        n = 12

                        # m = {'type': 'image', 'base64': [encoded_string[i:i + n] for i in range(0, len(encoded_string), n)]}
                        m = {'type': 'image',
                             'base64': encoded_string}
                        # print(len(m.get('base64')))
                else:
                    encoded_string = None
                    m = {'type': 'image',
                         'base64': 'None'}
                    print(m.get('base64'))
                return Response(encoded_string)

        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        K.clear_session()


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    depth = args.depth
    width = args.width

    # face = FaceCV(depth=depth, width=width)
    #
    # face.detect_face()