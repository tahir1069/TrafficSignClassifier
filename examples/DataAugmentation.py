#Some Usefull Functions
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os 
import matplotlib.image as mpimg
IMAGE_SIZE = 51


def show_imgs(source_img, source_img1, source_img2):
    plt.figure(figsize=(6, 2))
    plt.subplot(1, 3, 1)
    plt.imshow(source_img)
    plt.title("Raw image")
    plt.subplot(1, 3, 2)
    plt.imshow(source_img1)
    plt.title("Processed image")
    plt.subplot(1, 3, 3)
    plt.imshow(source_img2)
    plt.title("Augmented image")
    plt.show()
    print(source_img.shape)
    print(source_img1.shape)
    print(source_img2.shape)


def crop_img(source_img):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE),
                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Each image is resized individually as different image may be of different size.
        img = source_img  # Do not read alpha channel.
        resized_img = sess.run(tf_img, feed_dict={X: img})
    resized_img = np.array(resized_img, dtype=np.uint8)  # Convert to numpy
    return resized_img


def sharpen_img(source_img):
    gb = cv2.GaussianBlur(source_img, (5, 5), 0)
    return cv2.addWeighted(source_img, 2, gb, -1, 0)


def scale_img(source_img):
    img2 = source_img.copy()
    sc_y = 0.4 * np.random.rand() + 1.0
    img2 = cv2.resize(source_img, None, fx=1, fy=sc_y, interpolation=cv2.INTER_CUBIC)
    c_x, c_y, sh = int(img2.shape[0] / 2), int(img2.shape[1] / 2), int(img2.size / 2)
    return source_img[(c_x - sh):(c_x + sh), (c_y - sh):(c_y + sh)]


def rotate_img(source_img):
    c_x, c_y = int(source_img.shape[0] / 2), int(source_img.shape[1] / 2)
    ang = 40.0 * np.random.rand() - 20
    Mat = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
    return cv2.warpAffine(source_img, Mat, source_img.shape[:2])


def increase_brightness_img(source_img, value=5):
    hsv = cv2.cvtColor(source_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def contrast_img(source_img):
    lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def translate_img(source_img):
    rows, cols, _ = source_img.shape
    # allow translation up to px pixels in x and y directions
    px = 6
    dx, dy = np.random.randint(-px, px, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(source_img, M, (cols, rows))
    return dst


def transform_img(source_img):
    return sharpen_img(
        contrast_img(
            increase_brightness_img(
                crop_img(
                    source_img
                ))))


def augment_img(source_img):
    return transform_img(
        scale_img(
            translate_img(
                rotate_img(
                    source_img
                ))))
    
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if filename == 'Thumbs.db':
            pass
        else:
            img = transform_img(crop_img(mpimg.imread(os.path.join(folder,filename))))
            images.append(img)
    return images