import torch
from torch import nn
import torchvision
import torchvision.transforms as T
import timm
from cv2 import imread
from cv2 import resize as imresize  # cv2.resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # resnet = torchvision.models.resnet101(
        #    pretrained=True)  # pretrained ImageNet ResNet-101
        # weights=ResNet101_Weights.DEFAULT
        resnet = torchvision.models.resnet101(
            weights='ResNet101_Weights.DEFAULT')
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Batchnormalization

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))
        # self.batchnorm = nn.BatchNorm2d(
        #    (encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(
            images)  # (batch_size, 2048, image_size/32, image_size/32)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # out = self.batchnorm(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:  # was 5: previously
            for p in c.parameters():
                p.requires_grad = fine_tune

class EncoderMLPMixer(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(EncoderMLPMixer, self).__init__()
        self.enc_image_size = encoded_image_size
        self.resizeImg = T.Resize((224,224))
        #self.resizeImg = T.Compose([
        #                    T.Resize((224,224)),
                            #T.CenterCrop(224),
        #                    T.ToTensor(),
        #                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #                    ])
        #data_config = timm.data.resolve_model_data_config(model)
        #transforms = timm.data.create_transform(**data_config, is_training=False)
        self.mlpmixer = timm.create_model(
              'mixer_l16_224.goog_in21k_ft_in1k',
              pretrained=True,
              #img_size=256,
              num_classes=0,  # remove classifier nn.Linear
              pretrained_cfg_overlay={'mean': (0., 0., 0.), 
                          'std': (1., 1., 1.),}) # pretrained ImageNet MLPMixer

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        #The input for the EncoderCNN is 256x256
        #The input for MLPMixer is 224x224
        #The output for EncoderCNN is 14x14x2048
        #The output for MLPMixer is 


        #images = imresize(images, (256, 256))
        #img = img.transpose(2, 0, 1)
        #img = img / 255.
        #img = torch.FloatTensor(img).to(device)
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                std=[0.229, 0.224, 0.225])
        #transform = transforms.Compose([normalize])
        #image = transform(img)  # (3, 256, 256)
        
        images=self.resizeImg(images)
        out = self.mlpmixer.forward_features(images)  # (batch_size, 196, 1024)
        #(Batchsize, 196, 1024) shaped tensor
        #print(out.size()) #128,196,1024
        out = out.view(-1, 14, 14, 1024)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        for p in self.mlpmixer.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune last 8 mixer block out 24 and the layers after mlp mixer blocks
        for c in list(self.mlpmixer.children())[2:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
        for c in list(self.mlpmixer.children())[1][16:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

