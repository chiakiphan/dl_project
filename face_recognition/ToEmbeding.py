from facenet_pytorch import InceptionResnetV1
import torch


class ToEmbeding(object):
    def __call__(self, img):
        img = img.unsqueeze(0)
        device = torch.device('cpu')
        embed = InceptionResnetV1(pretrained='vggface2').eval().to(device=device)
        face_embed = embed(img.to(device)).detach().cpu()
        return face_embed

