

from models.networks.resnetUnet import ResNetUNet
from .img_segmentor_model import ImgSegmentor

# zhjd:原程序没用上
def get_img_segmentor_from_options(options):
    return ImgSegmentor(segmentation_model=ResNetUNet(n_channel_in=3, n_class_out=options.n_object_classes),
                        loss_scale=options.img_segm_loss_scale)