# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data.dataloader  import default_collate

from caching_dictionary import CachingDictionary

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders"""

    @staticmethod
    def collate_corresp(batch):
        """Collates everything other than correspondences in default manner.
        Correspondences are passed as list of points per key.
        """
        keys_of_interest = [
            ('points1', 0, -1),
            ('points2', 0, -1),
            ('points1', 0, 1),
            ('points2', 0, 1),
        ]

        corresp_dict = {k:[] for k in keys_of_interest}
        for inp in batch:
            for k in keys_of_interest:
                if k not in inp.keys():
                    continue
                corresp_dict[k].append(inp[k])
                del inp[k]

        # create equal sized batch
        min_corresp = 1000
        dict_keys = list(corresp_dict.keys())
        for img in (0,2):
            for i in range(len(corresp_dict[dict_keys[img]])):
                min_corresp = min(min_corresp, corresp_dict[dict_keys[img]][i].shape[1])

            for i in range(len(corresp_dict[dict_keys[img]])):
                corresp_dict[dict_keys[img]][i] = corresp_dict[dict_keys[img]][i][:, :min_corresp, :]
                corresp_dict[dict_keys[img + 1]][i] = corresp_dict[dict_keys[img]][i][:, :min_corresp, :]
            
        return default_collate(batch), corresp_dict


    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 load_corresp=False,
                 corresp_cache_path=None,
                 corresp_n=-1):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.to_image = transforms.ToPILImage()

        # Adding SIFT feature information.
        self.load_corresp = load_corresp
        self.corresp_cache_path = corresp_cache_path
        self.corresp_n = corresp_n

        assert not load_corresp or corresp_cache_path is not None, 'Please improve!'

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

        if self.load_corresp:
            self.corresp_dict = CachingDictionary(
                self.corresp_cache_path,
                self.load_correspondences,
            )

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # Doing this in (source scale) resolution and then normalizing the values.
        if self.load_corresp:
            image_t = self.to_image(inputs[("color", 0, 0)])
            image_t_prev = self.to_image(inputs[("color", -1, 0)])
            image_t_next = self.to_image(inputs[("color", 1, 0)])

            key1 = (index, frame_index - 1, frame_index)
            key2 = (index, frame_index, frame_index + 1)

            corresp_prev_t = self.corresp_dict.try_get(key1, image_t_prev, image_t)
            corresp_t_next = self.corresp_dict.try_get(key2, image_t, image_t_next)

            preproc_corresp = lambda x: self.to_tensor(
                x if self.corresp_n == -1 else x[: self.corresp_n]
            )

            # (t, t - 1) and (t, t + 1)
            inputs[('points1', 0, -1)] = preproc_corresp(corresp_prev_t[1])
            inputs[('points2', 0, -1)] = preproc_corresp(corresp_prev_t[0])
            inputs[('points1', 0, 1)] = preproc_corresp(corresp_t_next[0])
            inputs[('points2', 0, 1)] = preproc_corresp(corresp_t_next[1])

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def load_correspondences(self, image1, image2):
        raise NotImplementedError