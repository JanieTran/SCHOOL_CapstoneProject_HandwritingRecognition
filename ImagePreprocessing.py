from PIL import Image, ImageEnhance, ImageFilter


class ImagePreprocessing:
    def __init__(self):
        pass

    @staticmethod
    def image_enhance(image, brightness, contrast):
        image = ImageEnhance.Brightness(image).enhance(factor=brightness)
        image = ImageEnhance.Contrast(image).enhance(factor=contrast)
        return image

    @staticmethod
    def image_filter(image):
        # image = image.filter(ImageFilter.BLUR)
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SMOOTH)
        return image

    def enhance_and_filter(self, image, brightness, contrast):
        image = self.image_enhance(image=image, brightness=brightness, contrast=contrast)
        image = self.image_filter(image)
        return image
