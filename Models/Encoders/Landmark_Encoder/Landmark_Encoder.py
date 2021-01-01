import torch
import cv2

from common.utils import BBox, drawLandmark_multiple
from models.mobilefacenet import MobileFaceNet
from Retinaface import Retinaface


class Encoder_Landmarks(torch.nn.Module):
    def __init__(self):
        super(Encoder_Landmarks, self).__init__()
        self.model = MobileFaceNet([112, 112], 136)

        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.model = self.model.eval()
        # if torch.cuda.device_count() > 0:
        #     self.model = self.model.to("cuda")

    def forward(self, data):
        return self.model(data)

    def loss(self, input_attr_lnd, output_lnd):
        loss = torch.norm(input_attr_lnd - output_lnd, p=2)
        return loss

    def get_landmarks(self, img_path):
        img = cv2.imread(img_path)

        # face detector
        retinaface = Retinaface.Retinaface()
        faces = retinaface(img)
        if len(faces) == 0:
            print('NO face is detected!')
            return

        for face in faces:
            # get face img and box
            cropped_face, new_bbox = get_cropped_and_box(img, face)
            if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                continue

            test_face = cropped_face.copy()
            test_face = test_face / 255.0
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape((1,) + test_face.shape)
            input = torch.from_numpy(test_face).float()
            input = torch.autograd.Variable(input)

            output =  self.model(input)
            landmark = output[0].cpu().data.numpy()

            landmark = landmark.reshape(-1, 2)
            landmark = new_bbox.reprojectLandmark(landmark)
            img_with_lnd = drawLandmark_multiple(img, new_bbox, landmark)
        return landmark, img_with_lnd

def get_cropped_and_box(img, face, out_size = 112):
    height, width, _ = img.shape
    x1=face[0]
    y1=face[1]
    x2=face[2]
    y2=face[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h])*1.2)
    cx = x1 + w//2
    cy = y1 + h//2
    x1 = cx - size//2
    x2 = x1 + size
    y1 = cy - size//2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    print(x1, x2, y1, y2)
    new_bbox = list(map(int, [x1, x2, y1, y2]))
    new_bbox = BBox(new_bbox)
    cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
      cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    cropped_face = cv2.resize(cropped, (out_size, out_size))
    return cropped_face, new_bbox
