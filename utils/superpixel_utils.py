import numpy as np
from PIL import Image
import os



def read_superpixel():
    location = "/media/jiyoungkim/D84F-7233/supersupersuper/8b30crf"
    file_list = os.listdir(location)
    # for file in file_list:
    im = np.array(Image.open(os.path.join(location, '000000000009.png')))

    print(im.shape)
    print(im)

read_superpixel()