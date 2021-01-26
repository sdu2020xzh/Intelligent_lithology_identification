from frcnn import FRCNN
from PIL import Image
from tqdm import tqdm

frcnn = FRCNN()
image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/" + image_id + ".jpg"
    image = Image.open(image_path)
    image = frcnn.detect_image(image)
    image.save("./img/" + image_id + ".jpg")

print("Conversion completed!")