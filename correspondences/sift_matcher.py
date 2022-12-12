import cv2
import numpy as np
from correspondences.correspondence_matcher import CorrespondenceMatcher

class SIFTMatcher(CorrespondenceMatcher):
    """Correspondence generator using SIFT features."""

    def __init__(self, opt=None) -> None:
        super().__init__()

        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    def get_correspondences(self, img1, img2):
        kp1, des1 = self.sift.detectAndCompute(img1,None)
        kp2, des2 = self.sift.detectAndCompute(img2,None)

        matches = self.bf.knnMatch(des1, des2, k=2)

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

        return points1, points2
