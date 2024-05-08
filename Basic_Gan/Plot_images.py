import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
# 假定 fake_images, n_classes, n_examples 已经定义
def show_images(images, num_images, nrow=10, names=None, name_scale=1):
    plt.figure(figsize=(15 / 10 * nrow, 15))
    images = make_grid(images.cpu()[:num_images], nrow=nrow, padding=2, pad_value=1)
    images =  np.clip(images, 0, 1)
    npimg = images.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    single_img_height = npimg.shape[1] // (num_images // nrow) * name_scale
    if names is not None:
        for i, name in enumerate(names):
            if i >= num_images:
                break  # Break if there are more names than images
            row = i // nrow - 1
            col = i % nrow
            plt.text(col * single_img_height + single_img_height / 2, (row + 1) * single_img_height - single_img_height * 0.14,
                     name, ha='center', va='top', fontsize=12, color='black',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='white', boxstyle='round,pad=0.1'))
    plt.show()
show_images(fake_images/ 2 + 0.5, 20)