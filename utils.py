import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


def getLandmarks(image_name):
    mp_face_mesh = mp.solutions.face_mesh
    image = cv2.imread(image_name)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    try:
        image.flags.writeable == False
    except AttributeError:
        raise AttributeError("Can't read image")

    results = face_mesh.process(image)
    landmarks = results.multi_face_landmarks[0].landmark

    return landmarks


# getJoint take image RGB massive, joint_landmarks as list of landmarks
def getJoint(img, landmarks, joint_landmarks, bound_landmarks, deleteJoint=False):
    coordinates_list = []
    # Extract x,y coordinates of landmarks
    for i in joint_landmarks:
        landmark_coordinate = []

        x = int(landmarks[i].x * img.shape[1])
        landmark_coordinate.append(x)

        y = int(landmarks[i].y * img.shape[0])
        landmark_coordinate.append(y)

        coordinates_list.append(landmark_coordinate)

    coordinates = np.array(coordinates_list)
    # Create mask and define polygon
    mask = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillConvexPoly(mask, coordinates, 1)
    mask = mask.astype(np.bool)

    # Delete masked part from input image
    if deleteJoint:
        img[mask] = np.clip(img[mask], 255, 255)
    # Extract and return masked part from input image
    else:
        out = np.full(img.shape, 255)
        out[mask] = img[mask]
        return out[bound_landmarks[0]:bound_landmarks[1], bound_landmarks[2]:bound_landmarks[3]]


def getLips(image, landmarks, delete_joint=False):
    # Wide coordinates of joint
    lips_top = int(landmarks[164].y * image.shape[0])
    lips_left = int(landmarks[57].x * image.shape[1])
    lips_bottom = int(landmarks[18].y * image.shape[0])
    lips_right = int(landmarks[287].x * image.shape[1])

    bound_landmarks = [lips_top, lips_bottom, lips_left, lips_right]

    # Define joint bounding coordinates list
    lips_landmarks = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]

    if delete_joint:
        getJoint(image, landmarks, lips_landmarks, bound_landmarks, deleteJoint = True)
    else:
        return getJoint(image, landmarks, lips_landmarks, bound_landmarks)


def getNose(image, landmarks, delete_joint=False):
    # Wide coordinates of joint
    nose_top = int(landmarks[197].y * image.shape[0])
    nose_left = int(landmarks[102].x * image.shape[1])
    nose_bottom = int(landmarks[164].y * image.shape[0])
    nose_right = int(landmarks[331].x * image.shape[1])

    bound_landmarks = [nose_top, nose_bottom, nose_left, nose_right]

    # Define joint bounding coordinates list
    nose_landmarks = [2, 326, 460, 439, 360, 363, 456, 351, 6, 122, 236, 134, 131, 219, 240, 97]

    if delete_joint:
        getJoint(image, landmarks, nose_landmarks, bound_landmarks, deleteJoint=True)
    else:
        return getJoint(image, landmarks, nose_landmarks, bound_landmarks)


def getRightEye(image, landmarks, delete_joint=False):
    # Wide coordinates of joint
    eye_top = int(landmarks[257].y * image.shape[0])
    eye_left = int(landmarks[464].x * image.shape[1])
    eye_bottom = int(landmarks[450].y * image.shape[0])
    eye_right = int(landmarks[359].x * image.shape[1])

    bound_landmarks = [eye_top, eye_bottom, eye_left, eye_right]

    # Define joint bounding coordinates list
    right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    if delete_joint:
        getJoint(image, landmarks, right_eye_landmarks, bound_landmarks, deleteJoint = True)
    else:
        return getJoint(image, landmarks, right_eye_landmarks, bound_landmarks)


def getLeftEye(image, landmarks, delete_joint = False):
    # Wide coordinates of joint
    eye_top = int(landmarks[27].y * image.shape[0])
    eye_left = int(landmarks[130].x * image.shape[1])
    eye_bottom = int(landmarks[230].y * image.shape[0])
    eye_right = int(landmarks[244].x * image.shape[1])

    bound_landmarks = [eye_top, eye_bottom, eye_left, eye_right]

    # Define joint bounding coordinates list
    left_eye_landmarks = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]

    if delete_joint:
        getJoint(image, landmarks, left_eye_landmarks, bound_landmarks, deleteJoint=True)
    else:
        return getJoint(image, landmarks, left_eye_landmarks, bound_landmarks)


def getFace(image, landmarks):
    # Wide coordinates of joint
    face_top = int(landmarks[10].y * image.shape[0])
    face_left = int(landmarks[234].x * image.shape[1])
    face_bottom = int(landmarks[152].y * image.shape[0])
    face_right = int(landmarks[454].x * image.shape[1])

    bound_landmarks = [face_top, face_bottom, face_left, face_right]

    # Define joint bounding coordinates list
    face_landmarks = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332,
                      297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148]

    return getJoint(image, landmarks, face_landmarks, bound_landmarks)


# Get face without lips, nose and eyes
def getOnlyFace(image, landmarks):
    # Copy image array to avoid redefining original image
    image = image.copy()
    # Delete all face joints
    getLips(image, landmarks, delete_joint=True)
    getNose(image, landmarks, delete_joint=True)
    getRightEye(image, landmarks, delete_joint=True)
    getLeftEye(image, landmarks, delete_joint=True)

    result = getFace(image, landmarks)
    return result

def faceCutter(image_rout):
    # Create folder for croped parts
    image_name = image_rout.split('/')[-1]

    if os.path.exists(image_name):
        shutil.rmtree(image_name)
    os.makedirs(image_name)

    # Create landmarks and read image
    landmarks = getLandmarks(image_rout)
    image = cv2.imread(image_rout)

    # Crop and save face
    face = getFace(image, landmarks)
    cv2.imwrite(image_name + '/face.jpg', face)

    # Crop and save lips
    lips = getLips(image, landmarks)
    cv2.imwrite(image_name + '/lips.jpg', lips)

    # Crop and save nose
    nose = getNose(image, landmarks)
    cv2.imwrite(image_name + '/nose.jpg', nose)

    # Crop and save right eye
    right_eye = getRightEye(image, landmarks)
    cv2.imwrite(image_name + '/right_eye.jpg', right_eye)

    # Crop and save left eye
    left_eye = getLeftEye(image, landmarks)
    cv2.imwrite(image_name + '/left_eye.jpg', left_eye)


def addBrightness(input_path, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # Create filelist
    file_list = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    for file in file_list:
        try:
            landmarks = getLandmarks(input_path + '/' + file)
        except:
            continue
        img = cv2.imread(input_path + '/' + file)
        # Get face and extract mean brightness value from HSV
        face = getOnlyFace(img, landmarks)
        HSV = cv2.cvtColor(np.float32(face), cv2.COLOR_RGB2HSV)
        val = HSV[:, :, 2].mean()
        # Save image with new name
        cv2.imwrite(output_path +"/"+ str(int(val)) + "_" + file + ".jpg", img)