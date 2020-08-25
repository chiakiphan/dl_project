import os
import pandas as pd


def create_data(path2data='/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/video'
                          '/train', labels=-1):
    labels_data = []
    paths = []
    if type(labels) is not list:
        labels = [i for i in range(labels)]
    for idx, name in enumerate(sorted(os.listdir(path2data))):
        label = labels[idx]
        path = os.path.join(path2data, name)
        for img in sorted(os.listdir(path)):
            labels_data.append(label)
            paths.append(os.path.join(path, img))

    df = pd.DataFrame({'path': paths, 'label': labels_data})
    return df


def get_name(path2data='/home/kpst/PycharmProjects/face_classify_torch/face_detection_opencv/face_matching/video'
                       '/train', ):
    return sorted(os.listdir(path2data))
