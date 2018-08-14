
import cv2  
from PIL import Image  
from skimage import io,data
from matplotlib import pyplot as plt
import numpy  
from imgaug import augmenters as iaa
from torchvision import transforms
  
# img = cv2.imread("/home/zengh/Dataset/zhSensitive/Normal/dailylife/dailylife_000038.jpg")  
# cv2.imshow("OpenCV",img)  
# image = Image.open('/home/zengh/Dataset/zhSensitive/Normal/dailylife/dailylife_000034.jpg')
# image.show()  
img = io.imread('/home/zengh/Dataset/oxy_all/oxySensitive/Sensitive_train_img/s00001_Body_Half/s00001_00043.jpg')
seq = iaa.Sequential([
    # iaa.Crop(px=(0, 100)), # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(1), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(2, 3.0)), # blur images with a sigma of 0 to 3.0
    # iaa.Dropout(0.01),
    # iaa.Sharpen(0.2),
    # iaa.Affine(shear=15)
])
# skiimg = data.coins()
# print(skiimg)
h,w,c = img.shape
img = img.reshape((1,h,w,c))
images_aug = seq.augment_images(img)
images_aug = images_aug.reshape((h,w,c))
pil_img = transforms.ToPILImage()(images_aug).convert('RGB')
# print(type(pil_img))
pil_img.show()
# io.imshow(images_aug)

# cv2.waitKey() 
# plt.show() 