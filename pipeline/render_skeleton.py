
from algorithms.penpositions_to_skeletonimages import penpositions_to_skeletonimages_without_metadata

from PIL import Image, ImageFilter, ImageOps


def blur_skeleton(skeletonImg):
    return skeletonImg.filter(ImageFilter.GaussianBlur(1))

def render_skeleton(penPositions, imgSize=None, imgOffset=None):

    skeletonImg, pos = penpositions_to_skeletonimages_without_metadata(penPositions, imgOffset=imgOffset, imgSize=imgSize)
    
    skeletonImg = skeletonImg.convert('L')
    
    blurredSkeletonImg = blur_skeleton(ImageOps.invert(skeletonImg)).convert('RGB')

    return blurredSkeletonImg, skeletonImg