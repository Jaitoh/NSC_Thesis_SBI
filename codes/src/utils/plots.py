import cv2


def load_img(
    img_path,
    ax,
    title,
    crop=False,
    x_start=None,
    x_end=None,
    y_start=None,
    y_end=None,
    print_shape=False,
):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # print total number of pixels
    if print_shape:
        print(f"Image number of pixels: {img.shape[0], img.shape[1]}")

    # Crop image
    if crop:
        img = img[y_start:y_end, x_start:x_end]

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
