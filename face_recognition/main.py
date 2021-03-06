import torch
import cv2
import os
from facenet_pytorch import MTCNN
from train import FaceClassify, get_normalize
from dataset import get_name
from PIL import Image
from skimage.measure import compare_ssim
from sklearn.preprocessing import Normalizer
import pickle
import numpy as np


def draw_bbox(image, coord, color, text=''):
    x, y, w, h = coord
    image = cv2.rectangle(image,
                          (x, y),
                          (x + w, y + h),
                          color=color,
                          thickness=2)
    image = cv2.putText(image, str(text), (x + 10, y + h + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, [0, 255, 0], 1)
    return image


def get_face(image, thresh_hold=0.75):
    bbox, conf = MTCNN().detect(image)
    if bbox is None:
        bbox, conf = [], []
    for b, c in zip(bbox, conf):
        if c < thresh_hold:
            continue
        x1, y1, x2, y2 = b.astype(int)
        yield image[max(y1, 0):y2, max(x1, 0):x2], (x1, y1, x2 - x1, y2 - y1)


def get_feature(image):
    image = Image.fromarray(image)
    normalize = get_normalize()
    feature = normalize(image)
    return feature


def load_model(model_save):
    model = FaceClassify()
    model.load_state_dict(torch.load(model_save))
    model.eval()
    return model


def load_model_svm(model_save):
    model = pickle.load(open(model_save, 'rb'))
    return model


def get_feature_svm(image):
    feature = get_feature(image).numpy()
    feature = Normalizer(norm='l2').transform(feature)
    return feature


def calculate_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist


def predict_video(link_video, model=None, speed=1, save_result=None):
    names = get_name()
    cap = cv2.VideoCapture(link_video)
    color = {4: [255, 0, 0], 3: [0, 0, 255], -1: [0, 255, 0]}
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(save_result, fourcc, 20.0, (640, 480))
    anchor = None
    bbox = []
    flag = True
    while cap.isOpened():
        ret, frame = cap.read()
        frame_rate = cap.get(1)
        frame = cv2.resize(frame, (640, 480))
        if flag:
            flag = False
            anchor = frame
            for idx, (face, coord) in enumerate(get_face(frame)):
                draw_bbox(frame, coord, [200, 0, 200], '')
                if coord[2] + coord[3] < 90:
                    continue
                if idx == 0:
                    bbox = []
                    anchor = frame
                acc, label = predict_image(face, model)
                # acc, label = predict_image_svm(face, model)
                cv2.putText(frame, '%.2f | %d | %d' % (acc, label, frame_rate), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            [0, 255, 0],
                            2)
                if label in [3, 4]:
                    print('acc %.3f %d' % (acc, frame_rate))
                    if acc > 0.5:
                        draw_bbox(frame, coord, color[label], text=names[label])
                        bbox.append((label, coord))
            # print('anchor {}'.format(frame_rate))
        else:
            score = compare_ssim(cv2.cvtColor(anchor[320:640, :], cv2.COLOR_BGR2GRAY),
                                 cv2.cvtColor(frame[320:640, :], cv2.COLOR_BGR2GRAY),
                                 full=False)
            # print('socre %.3f, %d' % (score, frame_rate))
            if score > 0.78:
                cv2.putText(frame, '{}'.format(frame_rate), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255],
                            2)
                if len(bbox) > 0:
                    for id, face in bbox:
                        draw_bbox(frame, face, color=color[id], text=names[id])
            else:
                bbox = []
                flag = True
        cv2.imshow("video", frame)
        if save_result:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_image(image, model):
    feature = get_feature(image)
    with torch.no_grad():
        predicted = model(feature)
        score = torch.nn.functional.softmax(predicted, 1)
        predicted = torch.squeeze(score, 1)
        score, predicted = torch.max(predicted.data, 1)
    return score.item(), predicted.item()


def predict_image_svm(image, model):
    feature = get_feature_svm(image)
    pred = model.predict_proba(feature)
    return np.max(pred, 1).item(), np.argmax(pred, 1).item()


def ssmi(img1, img2):
    score = compare_ssim(img1, img2)
    return score


def main():
    model_save = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/deeplearning' \
                 '/model_checkpoint/model_6.321091677993536.ckpt'
    svm_model = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/deeplearning' \
                '/model_checkpoint/model.ml'
    video_link = '/home/kpst/Downloads/test_video.mp4'
    save = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/video/output.avi'
    model = load_model(model_save)
    predict_video(video_link, model, speed=1, save_result=save)
    # svm = load_model_svm(svm_model)
    # predict_video(video_link, svm, speed=1, save_result=save)
    # img = cv2.imread('/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/video/test'
    #                  '/1338.0.png')
    # for face, _ in get_face(img):
    #     s, a = predict_image(face, model)


if __name__ == '__main__':
    main()

    # from facenet_pytorch import MTCNN
    # import time
    # import matplotlib.pyplot as plt
    # anchor = cv2.imread(
    #     '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/video/test/1338.0.png', 0)
    # s = time.time()
    # for i in range(10, 28):
    #     link = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/video/test/29{}.0.png'.format(
    #         i)
    #
    #     img = cv2.imread(link)
    #     # img = img[320:640, :]
    #     # print(ssmi(anchor[320:640, :], img))
    # e = time.time()
    # print('time: {}'.format((e - s) * 1000))
