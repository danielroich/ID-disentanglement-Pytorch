import torch
import cv2

from .Facenet.utils import BBox, drawLandmark_only
from .Facenet.mobilefacenet import MobileFaceNet
from .Retinaface.Retinaface import Retinaface


class Encoder_Landmarks(torch.nn.Module):
    def __init__(self, model_dir='Weights/mobilefacenet_model_best.pth.tar',
                 retinaface_model_dir='Weights/mobilenet0.25_Final.pth'):
        super(Encoder_Landmarks, self).__init__()
        self.model = MobileFaceNet([112, 112], 136)

        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.retinaface_model_dir = retinaface_model_dir

        # self.model = self.model.train()
        # if torch.cuda.device_count() > 0:
        #     self.model = self.model.to("cuda")

    # assume imgs is np in shape (batch_size, 256, 256, 3)
    def forward(self, imgs):
        # preprocess - get input and face boxes
        inputs, boxes = self.get_inputs_and_boxes(imgs)
        inputs = torch.autograd.Variable(inputs)

        # pass our model as a batch
        outputs, _ = self.model(inputs)

        # postprocess
        landmarks = self.reproject_landmarks(boxes, outputs)

        return outputs, landmarks

    def loss(self, input_attr_lnd, output_lnd):
        loss = torch.norm(input_attr_lnd - output_lnd, p=2)
        return loss

    # postprocess
    def reproject_landmarks(self, boxes, landmarks):
        landmarks_ = torch.clone(landmarks)
        batch_size = landmarks_.shape[0]
        landmarks_ = torch.reshape(landmarks_, (batch_size, 68, 2))
        for i in range(batch_size):
            landmarks_[i, :, 0] = landmarks_[i, :, 0] * boxes[i].w + boxes[i].x
            landmarks_[i, :, 1] = landmarks_[i, :, 1] * boxes[i].h + boxes[i].y
        return landmarks_

    # preprocess
    # return inputs (batch_size, 3, 112, 112), BBox list in length batch_size
    def get_inputs_and_boxes(self, imgs, out_size=112):
        inputs = torch.zeros(imgs.shape[0], imgs.shape[3], out_size, out_size)
        boxes = []
        for i, img in enumerate(imgs):

            retinaface = Retinaface(trained_model=self.retinaface_model_dir)
            faces = retinaface(img)
            if len(faces) == 0 or len(faces) > 1:
                print('NO faces or more than one face is detected!')
                return

            face = faces[0]

            # get face img and box
            cropped_face, new_bbox = self.get_cropped_and_box(img, face)
            if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
                print('Error in cropped face!')
                return

            test_face = cropped_face.copy()
            test_face = test_face / 255.0
            test_face = test_face.transpose((2, 0, 1))
            test_face = test_face.reshape(test_face.shape)
            input_ = torch.from_numpy(test_face).float()
            # input = torch.autograd.Variable(input, requires_grad=True)

            inputs[i] = input_
            boxes.append(new_bbox)  # append to end of list
        return inputs, boxes

    # part of preprocess
    def get_cropped_and_box(self, img, face, out_size=112):
        height, width, _ = img.shape
        x1 = face[0]
        y1 = face[1]
        x2 = face[2]
        y2 = face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h]) * 1.2)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        cropped_face = cv2.resize(cropped, (out_size, out_size))
        return cropped_face, new_bbox
