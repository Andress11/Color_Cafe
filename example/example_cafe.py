from Cafe_Color.read_features import Image
from Cafe_Color.preprocessing import Preprocess
from Cafe_Color.segmentation import ColorSegmentation

import matplotlib.pyplot as plt

if __name__ == '__main__':

    img = Image('cafe2.jpg')
    img_normal = Preprocess(img)._normalize()
    Color = ColorSegmentation(img_normal)
    results = Color.MaskLab_coffe(((22,99),(15,100)))

    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    plt.title('Café Bueno: '+ str(results.percent[0]))
    plt.imshow(results.good_sample)

    plt.subplot(1,2,2)
    plt.title('Café Malo: '+ str(results.percent[1]))
    plt.imshow(results.bad_sample)
    plt.show()