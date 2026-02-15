import argparse
import math

import cv2
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("The transformers package is not installed. Please install it to use CLIP.")
    exit(1)


from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='test/3710674892.jpg',
        help='Input image path')
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=["a cat", "a dog", "a car", "a person", "a shoe", 'an apple', 'a boy'],
        help='need recognition labels'
    )

    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=None, width=None):
    token_count = tensor.size(1) - 1  # exclude CLS token
    if height is None or width is None:
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"Unexpected token count for ViT map: {token_count}")
        height = side
        width = side

    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class ImageClassifier(nn.Module):
    def __init__(self, labels):
        super(ImageClassifier, self).__init__()
        self.clip = CLIPModel.from_pretrained("/datanfs4/shenruoyan/checkpoints/clip/ViT-L-14-336")
        self.processor = CLIPProcessor.from_pretrained("/datanfs4/shenruoyan/checkpoints/clip/ViT-L-14-336")
        self.image_size = self.clip.config.vision_config.image_size
        self.labels = labels

    def forward(self, x):
        text_inputs = self.processor(text=self.labels, return_tensors="pt", padding=True)

        outputs = self.clip(pixel_values=x, input_ids=text_inputs['input_ids'].to(self.clip.device),
                            attention_mask=text_inputs['attention_mask'].to(self.clip.device))

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        for label, prob in zip(self.labels, probs[0]):
            print(f"{label}: {prob:.4f}")
        return probs


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    labels = args.labels
    model = ImageClassifier(labels).to(torch.device(args.device)).eval()
    print(model)

    target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (model.image_size, model.image_size))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]).to(args.device)

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)



    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    #targets = [ClassifierOutputTarget(1)]
    targets = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    img_name = args.image_path.split('/')[-1].split('.')[0]
    cv2.imwrite(f'test/{img_name}_cam.jpg', cam_image)
