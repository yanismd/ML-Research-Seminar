import matplotlib.pyplot as plt
from src.utils.utils import pytorch_to_numpy


def display_images(imgs):
    r = imgs.shape[0]//5
    c = 5
    fig, axs = plt.subplots(r, c)
    fig.suptitle("Generated data")
    k = 0
    if r == 1:
        for j in range(c):
            display_img = pytorch_to_numpy(imgs[j, :, :, :].permute(1, 2, 0))
            axs[j].imshow(display_img)
            axs[j].axis('off')

        plt.show()
        return

    for i in range(r):
        for j in range(c):
            display_img = pytorch_to_numpy(imgs[k, :, :, :].permute(1, 2, 0))
            axs[i][j].imshow(display_img)
            axs[i][j].axis('off')
            k += 1
    plt.show()


def display_restoration_process(
        target_data_list,
        altered_data_list,
        restored_data_list,
        max_samples=5
):
    n_cols = max_samples
    fig, axs = plt.subplots(nrows=3, ncols=n_cols)
    fig.suptitle("Target, Altered, Restored data")
    for j in range(n_cols):
        axs[0][j].imshow(pytorch_to_numpy(target_data_list[0][j]).squeeze())
        axs[1][j].imshow(pytorch_to_numpy(altered_data_list[0][j]).squeeze())
        axs[2][j].imshow(pytorch_to_numpy(restored_data_list[0][j]).squeeze())

        axs[0][j].axis('off')
        axs[1][j].axis('off')
        axs[2][j].axis('off')

    plt.show()
