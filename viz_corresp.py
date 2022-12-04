import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def get_sift_corresp(img1, img2):

    h, w = img1.shape[0], img2.shape[1]

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # Sorting by distance.
    good.sort(key=lambda x: x.distance)

    # Which one is query and which one is train.
    # https://github.com/opencv/opencv/blob/4.x/modules/features2d/src/draw.cpp#L239
    points1 = np.asarray([kp1[match.queryIdx].pt for match in good])
    points2 = np.asarray([kp2[match.trainIdx].pt for match in good])

    # # Normalize values
    # min_max = lambda x: 2 * x / np.array([[w, h]]) - 1

    # norm_points1 = min_max(points1)
    # norm_points2 = min_max(points2)

    return points1, points2

def draw_line(img, pts, color, thickness=3):
    cv2.line(
        img,
        [int(i) for i in pts[0]],
        [int(i) for i in pts[1]],
        color,
        thickness=thickness
    )

    return img

def draw_correspondences(img1, img2, points1, points2, color=[0, 255, 0]):

    assert img1.shape[0] == img2.shape[0]
    assert points1.shape[-1] == points2.shape[-1] == 2
    assert points1.shape == points2.shape

    combined_img = np.hstack((img1, img2))
    new_points2 = points2 + [img1.shape[1], 0] # Moving the x coordinate accordingly for img2.

    for pt1, pt2 in zip(points1, new_points2):
        combined_img = draw_line(combined_img, [pt1, pt2], color, thickness=3)

    return combined_img

def draw_three_corresp(img1, img2, img3, corresp12, corresp23, n):

    h, w = img1.shape[0], img2.shape[1]

    corr_slice = slice(0, n, 1)

    # Green
    img12 = draw_correspondences(
        img1,
        img2,
        corresp12[0][corr_slice],
        corresp12[1][corr_slice]
    )

    # Red
    img23 = draw_correspondences(
        img12,
        img3,
        corresp23[0][corr_slice] + np.array([[w, 0]]),
        corresp23[1][corr_slice],
        color=[255, 0, 0]
    )

    return img23

def main(args):
    path_t = args.path_t
    path_prev = args.path_prev
    path_next = args.path_next
    corresp_n = args.corresp_n

    orig_img_t = cv2.cvtColor(cv2.imread(path_t), cv2.COLOR_BGR2RGB)
    orig_img_prev = cv2.cvtColor(cv2.imread(path_prev), cv2.COLOR_BGR2RGB)
    orig_img_next = cv2.cvtColor(cv2.imread(path_next), cv2.COLOR_BGR2RGB)

    img_t = cv2.cvtColor(orig_img_t, cv2.COLOR_RGB2GRAY)
    img_prev = cv2.cvtColor(orig_img_prev, cv2.COLOR_RGB2GRAY)
    img_next = cv2.cvtColor(orig_img_next, cv2.COLOR_RGB2GRAY)

    corresp_prev_t = get_sift_corresp(img_prev, img_t)
    corresp_t_next = get_sift_corresp(img_t, img_next)

    combined_img = draw_three_corresp(
        orig_img_prev,
        orig_img_t,
        orig_img_next,
        corresp_prev_t,
        corresp_t_next,
        corresp_n,
    )

    plt.axis('off')
    plt.imshow(combined_img)
    plt.savefig(
        'corresp_result.png',
        bbox_inches='tight',
        dpi=200,
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path-t',
        help="Defines the file path to image at time stamp t.",
        type=str,
        default='img1.png'
    )
    parser.add_argument(
        '--path-prev',
        help="Defines the file path to image at time stamp t - 1.",
        type=str,
        default='img2.png'
    )
    parser.add_argument(
        '--path-next',
        help="Defines the file path to image at time stamp t + 1.",
        type=str,
        default='img3.png'
    )
    parser.add_argument(
        '--corresp-n',
        help="Defines the number of correspondences.",
        type=int,
        default=5
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    main(args)