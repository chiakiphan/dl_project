import torch
import cv2
import os
from facenet_pytorch import MTCNN
from train import FaceClassify, get_normalize
from dataset import get_name
from PIL import Image
from skimage.measure import compare_ssim


def draw_bbox(image, coord, color, text=''):
    x, y, w, h = coord
    image = cv2.rectangle(image,
                          (x, y),
                          (x + w, y + h),
                          color=color,
                          thickness=2)
    image = cv2.putText(image, str(text), (x + 10, y + h + 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, [0, 255, 0], 1)
    return image


def get_face(image, thresh_hold=0.98):
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


def calculate_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist


def predict_video(link_video, model=None, speed=1, save_result=None):
    names = get_name()
    cap = cv2.VideoCapture(link_video)
    color = {4: [255, 0, 0], 8: [0, 0, 255], 0: [0, 255, 0]}
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
            for idx, (_, coord) in enumerate(get_face(frame)):
                if idx == 0:
                    bbox = []
                    anchor = frame
                    flag = True
                draw_bbox(frame, coord, color[0])
                bbox.append(coord)
        else:
            (score, _) = compare_ssim(cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY),
                                      cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                      full=True)
            if score > 0.90:
                cv2.putText(frame, str(score), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, [0, 0, 255], 2)
                if len(bbox) > 0:
                    for c in bbox:
                        draw_bbox(frame, c, color=color[4])
            else:
                bbox = []
                flag = True
        cv2.imshow("video", frame)
        if save_result:
            writer.write(frame)
        # if frame_rate % speed == 0:

        # for face, coord in get_face(frame):

        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # feature = get_feature(face)
        # with torch.no_grad():
        #     label = model_checkpoint(feature)
        #     score = torch.nn.functional.softmax(label)
        #     _, predicted = torch.max(label.data, 1)
        #     predicted = predicted.item()
        #     print(score[0][predicted])
        #     # if predicted == 8 and score[0][predicted] < 0.92:
        #     #     continue
        #     # if score[0][predicted] < 0.8:
        #     #     continue
        # if predicted not in [4, 8]:
        #     # name = names[predicted]
        #     predicted = 0

        # draw_bbox(frame, coord, color[0])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_image(image, model):
    feature = get_feature(image)
    with torch.no_grad():
        label = model(feature)
        print(label)
        score = torch.nn.functional.softmax(label, 1)
        _, predicted = torch.max(label.data, 1)
    print(predicted, score)


def main():
    model_save = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/deeplearning' \
                 '/model_checkpoint/model=80.61224489795919.ckpt'
    video_link = '/home/kpst/Downloads/test_video.mp4'
    save = '/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/video/output_detect.avi'
    model = load_model(model_save)
    predict_video(video_link, model, speed=1, save_result=save)
    # img = cv2.imread('/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/image/lee bo '
    #                  'young/lee bo young37.png')
    # for face, _ in get_face(img):
    #     predict_image(face, model_checkpoint)


if __name__ == '__main__':
    main()
