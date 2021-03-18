import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from Configs import Global_Config

IMAGE_SIZE = 220
mtcnn = MTCNN(
    image_size=IMAGE_SIZE, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=Global_Config.device
)
to_pil = transforms.ToPILImage(mode='RGB')
crop_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                          transforms.CenterCrop(IMAGE_SIZE)])

resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(Global_Config.device)

class ID_Encoder(torch.nn.Module):

    def __init__(self):
        super(ID_Encoder, self).__init__()

    def crop_tensor_according_to_bboxes(self, images, bboxes):
        cropped_batch = []
        for idx, image in enumerate(images):
            try:
                cropped_image = crop_transform(image[:, int(bboxes[idx][0][1]):int(bboxes[idx][0][3]),
                                        int(bboxes[idx][0][0]):int(bboxes[idx][0][2])].unsqueeze(0))
            except:
                cropped_image = crop_transform(image.unsqueeze(0))
            cropped_batch.append(cropped_image)

        return torch.cat(cropped_batch, dim=0)

    def preprocess_images_to_id_encoder(self, images):
        bboxes = [mtcnn.detect(to_pil(image))[0] for image in images]
        cropped_images = self.crop_tensor_according_to_bboxes(images, bboxes)
        return cropped_images

    def forward(self, images):
        cropped_images = self.preprocess_images_to_id_encoder(images)
        img_embeddings = resnet(cropped_images)
        return img_embeddings