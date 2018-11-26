import torch
import os
import argparse
from src.model import AutoEncoder
from src.loader import load_images, normalize_images
from src.utils import initialize_exp, bool_flag, attr_flag
from torch.autograd import Variable

DATA_PATH = './data'


def get_params():
    parser = argparse.ArgumentParser(description='Images autoencoder')
    parser.add_argument("--images_filename", type=str, default="images_256_256.pth",
                        help="Images name")
    parser.add_argument("--ae", type=str, default="./models/best_accu_ae.pth",
                        help="Images name")
    parser.add_argument("--img_sz", type=int, default=256,
                        help="Image sizes (images have to be squared)")
    parser.add_argument("--img_fm", type=int, default=3,
                        help="Number of feature maps (1 for grayscale, 3 for RGB)")
    parser.add_argument("--attr", type=attr_flag, default="Race.5",
                        help="Attributes to classify")
    parser.add_argument("--instance_norm", type=bool_flag, default=False,
                        help="Use instance normalization instead of batch normalization")
    parser.add_argument("--init_fm", type=int, default=32,
                        help="Number of initial filters in the encoder")
    parser.add_argument("--max_fm", type=int, default=512,
                        help="Number maximum of filters in the autoencoder")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of layers in the encoder / decoder")
    parser.add_argument("--n_skip", type=int, default=0,
                        help="Number of skip connections")
    parser.add_argument("--deconv_method", type=str, default="convtranspose",
                        help="Deconvolution method")
    parser.add_argument("--hid_dim", type=int, default=512,
                        help="Last hidden layer dimension for discriminator / classifier")
    parser.add_argument("--dec_dropout", type=float, default=0.,
                        help="Dropout in the decoder")
    params = parser.parse_args()
    return params


def main():
    params = get_params()
    print('Loading images...')
    images = torch.load(os.path.join(DATA_PATH, params.images_filename))
    print('Normalizing images...')
    images_norm = normalize_images(images)

    print('Loading model...')
    ae = torch.load(params.ae).eval()
    print('Encoding images...')
    images_encoded = ae.encode(images_norm)

    print("Saving encoded images to %s ..." % DATA_PATH)
    torch.save(images_encoded[-1], 'images_512_4_4.pth')


if __name__ == '__main__':
    main()
