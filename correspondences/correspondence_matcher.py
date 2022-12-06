class CorrespondenceMatcher:
    """Superclass for feature extractor, descriptors and correspondence
    generators."""

    def get_correspondences(self, img1, img2):
        raise NotImplementedError