import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as VF

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(img_tensor, prob_tensor):
    batch_size = list(img_tensor.size())[0]
    img_tensor_cpu = img_tensor.detach().cpu()
    prob_tensor_cpu = prob_tensor.detach().cpu()
    out = []
    for i in range(batch_size):
        out_ = dense_crf(img_tensor_cpu[i], prob_tensor_cpu[i])
        out.append(out_)

    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in out], dim=0)
