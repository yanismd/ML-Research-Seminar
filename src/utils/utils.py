import matplotlib.pyplot as plt


def pytorch_to_numpy(x):
    return x.detach().numpy()


def display_images(imgs):
    r = 1
    c = imgs.shape[0]
    fig, axs = plt.subplots(r, c)
    for j in range(c):
        # black and white images
        axs[j].imshow(pytorch_to_numpy(imgs[j, 0, :, :]), cmap='gray')
        axs[j].axis('off')
    plt.show()
