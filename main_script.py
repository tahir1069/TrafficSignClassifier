# Load pickled data
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
import MyAlexNet
import DataAugmentation as func
import glob
import csv


# TODO: Fill this in based on where you saved the training and testing data

training_file = "train.p"
validation_file = "valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train, X_train_size, X_train_bbox = train['features'], train['labels'], train['sizes'], train['coords']
X_valid, y_valid, X_valid_size, X_valid_bbox = valid['features'], valid['labels'], valid['sizes'], valid['coords']
X_test, y_test, X_test_size, X_test_bbox = test['features'], test['labels'], test['sizes'], test['coords']

# TODO: Number of training examples
n_train = len(X_train_size)
# TODO: Number of validation examples
print(len(X_valid_size))
n_validation = len(X_valid_size)
# TODO: Number of testing examples.
n_test = len(X_test_size)
# TODO: What's the shape of an traffic sign image?
print(X_train.shape)
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

# TODO: Number of training examples
n_train = len(X_train_size)
# TODO: Number of testing examples.
n_test = len(X_test_size)
# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))
img_size = X_train.shape[1]  # Size of input images
print(img_size)
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization goes here.
# Visualizations will be shown in the notebook.

num_of_samples = []
plt.figure(figsize=(12, 16.5))
for i in range(0, n_classes):
    plt.subplot(11, 4, i + 1)
    x_selected = X_train[y_train == i]
    plt.imshow(x_selected[0, :, :, :])  # draw the first image of each class
    plt.title(i)
    plt.axis('off')
    num_of_samples.append(len(x_selected))
plt.show()

# Plot number of images per class
plt.figure(figsize=(12, 4))
plt.bar(range(0, n_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

print("Min number of images in training data per class =", min(num_of_samples))
print("Max number of images in training data per class =", max(num_of_samples))

### Data exploration visualization goes here.
# Visualizations will be shown in the notebook.

num_of_samples = []
plt.figure(figsize=(12, 16.5))
for i in range(0, n_classes):
    plt.subplot(11, 4, i + 1)
    x_selected = X_valid[y_valid == i]
    plt.imshow(x_selected[0, :, :, :])  # draw the first image of each class
    plt.title(i)
    plt.axis('off')
    num_of_samples.append(len(x_selected))
plt.show()

# Plot number of images per class
plt.figure(figsize=(12, 4))
plt.bar(range(0, n_classes), num_of_samples)
plt.title("Distribution of the validation dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

print("Min number of images in vlidation data per class =", min(num_of_samples))
print("Max number of images in validation data per class =", max(num_of_samples))

### Data exploration visualization goes here.
# Visualizations will be shown in the notebook.

num_of_samples = []
plt.figure(figsize=(12, 16.5))
for i in range(0, n_classes):
    plt.subplot(11, 4, i + 1)
    x_selected = X_test[y_test == i]
    plt.imshow(x_selected[0, :, :, :])  # draw the first image of each class
    plt.title(i)
    plt.axis('off')
    num_of_samples.append(len(x_selected))
plt.show()

# Plot number of images per class
plt.figure(figsize=(12, 4))
plt.bar(range(0, n_classes), num_of_samples)
plt.title("Distribution of the test dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

print("Min number of images in test data per class =", min(num_of_samples))
print("Max number of images in test data per class =", max(num_of_samples))


### For Data Augmentation

# X_train_aug = []
# y_train_aug = []
# def create_data(n):
#    for i in range(100):
#        img=X_train[i]
#        X_train_aug.append(img)
#        y_train_aug.append(y_train[i])
#        #Generate n new images out of each input image
#        for j in range(n):
#            X_train_aug.append(augment_img(img))
#            y_train_aug.append(y_train[i])


# X_train_crop = np.ndarray(shape=[X_train.shape[0],IMAGE_SIZE,IMAGE_SIZE,
#                                 3],dtype = np.uint8)
# for i in range(n_train):
#    X_train_crop[i] = crop_img(X_train[i])
#    print(i)

print(X_train.shape)
print(X_train.dtype)
print(y_train.shape)
print(y_train.dtype)

print(X_valid.shape)
print(X_valid.dtype)
print(y_valid.shape)
print(y_train.dtype)

print(X_test.shape)
print(X_test.dtype)
print(y_test.shape)
print(y_test.dtype)

filename = "updated_test.p"
file = open(filename, 'rb')
X_test = pickle.load(file)

filename = "updated_train.p"
file = open(filename, 'rb')
X_train = pickle.load(file)

filename = "updated_valid.p"
file = open(filename, 'rb')
X_valid = pickle.load(file)

test = X_train[10000]
transformation = func.transform_img(test)
augmentation = func.augment_img(test)
func.show_imgs(test, transformation, augmentation)


print(X_train.shape)
print(X_train.dtype)
print(y_train.shape)
print(y_train.dtype)

print(X_valid.shape)
print(X_valid.dtype)
print(y_valid.shape)
print(y_train.dtype)

print(X_test.shape)
print(X_test.dtype)
print(y_test.shape)
print(y_test.dtype)

# Data Normalization

print(np.mean(X_train))
X_train = (X_train - np.mean(X_train)) / 255.0
print(np.mean(X_train))

print(np.mean(X_valid))
X_valid = (X_valid - np.mean(X_valid)) / 255.0
print(np.mean(X_valid))

print(np.mean(X_test))
X_test = (X_test - np.mean(X_test)) / 255.0
print(np.mean(X_test))

## Shuffle the training dataset

print(X_train.shape)
print(y_train.shape)
X_train, y_train = shuffle(X_train, y_train)
print(X_train.shape)
print(y_train.shape)
print('done')

EPOCHS = 90
BATCH_SIZE = 128

print('done')

tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 51, 51, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)  # probability to keep units
one_hot_y = tf.one_hot(y, 43)

print('done')

rate = 0.0005
save_file = './new_model.ckpt'

logits = MyAlexNet.AlexNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

Saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('done')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        print("Epoch: ", i)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    Saver.save(sess,save_file)
    print("Model saved")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./Trained Model/final_model.ckpt.meta')
    saver2.restore(sess, "./Trained Model/final_model.ckpt")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))
    graph = tf.get_default_graph()

signs_class=[]
with open('signnames.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        signs_class.append((row['SignName']))
my_labels = [37,38,17,15,12,13,1,0,35,20,3,5]
test = func.load_images("./new_images1/")
test_images=X_test_data=np.uint8(np.zeros((len(test),51,51,3)))
test_images_labels=np.ndarray(shape=[len(test)],dtype=np.uint8)
test_images[0:12]=test[0:12]
test_images_labels[0:12]=my_labels[0:12]
plt.figure(figsize=(12, 8))
for i in range(len(test)):
    plt.subplot(3, 4, i+1)
    plt.imshow(test[i]) 
    plt.title(signs_class[my_labels[i]])
    plt.axis('off')
plt.show()
test_images=(test_images-np.mean(test_images))/255.0
### Visualize the softmax probabilities here.
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./Trained Model/final_model.ckpt.meta')
    saver2.restore(sess, "./Trained Model/final_model.ckpt")
    new_test_accuracy = evaluate(test_images, test_images_labels)
    print("New Test Set Accuracy = {:.3f}".format(new_test_accuracy))

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=5)
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./Trained Model/final_model.ckpt.meta')
    saver2.restore(sess, "./Trained Model/final_model.ckpt")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: test_images, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: test_images, keep_prob: 1.0})
print(len(test))
plt.figure(figsize=(16, 21))
for i in range(12):
    plt.subplot(12, 2, 2*i+1)
    plt.imshow(test[i]) 
    plt.title(i)
    plt.axis('off')
    plt.subplot(12, 2, 2*i+2)
    plt.barh(np.arange(1, 6, 1), my_top_k.values[i, :])
    labs=[signs_class[j] for j in my_top_k.indices[i]]
    plt.yticks(np.arange(1, 6, 1), labs)
plt.show()

my_labels = [3, 11, 1, 12, 38, 34, 18, 25]
test = []
for i, img in enumerate(glob.glob('./new_images2/*x.png')):
    image = func.crop_img(cv2.imread(img))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    test.append(image)
    
test_images=X_test_data=np.uint8(np.zeros((len(test),51,51,3)))
test_images_labels=np.ndarray(shape=[len(test)],dtype=np.uint8)
test_images[0:len(test)]=test[0:len(test)]
test_images_labels[0:len(test)]=my_labels[0:len(test)]
plt.figure(figsize=(12, 8))
for i in range(len(test)):
    plt.subplot(3, 4, i+1)
    plt.imshow(test[i]) 
    plt.title(signs_class[my_labels[i]])
    plt.axis('off')
plt.show()
test_images=(test_images-np.mean(test_images))/255.0
### Visualize the softmax probabilities here.
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./Trained Model/final_model.ckpt.meta')
    saver2.restore(sess, "./Trained Model/final_model.ckpt")
    new_test_accuracy = evaluate(test_images, test_images_labels)
    print("New Test Set Accuracy = {:.3f}".format(new_test_accuracy))

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=5)
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./Trained Model/final_model.ckpt.meta')
    saver2.restore(sess, "./Trained Model/final_model.ckpt")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: test_images, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: test_images, keep_prob: 1.0})
print(len(test))
plt.figure(figsize=(16, 21))
for i in range(len(test)):
    plt.subplot(12, 2, 2*i+1)
    plt.imshow(test[i]) 
    plt.title(i)
    plt.axis('off')
    plt.subplot(12, 2, 2*i+2)
    plt.barh(np.arange(1, 6, 1), my_top_k.values[i, :])
    labs=[signs_class[j] for j in my_top_k.indices[i]]
    plt.yticks(np.arange(1, 6, 1), labs)
plt.show()


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
#
#def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
#    # Here make sure to preprocess your image_input in a way your network expects
#    # with size, normalization, ect if needed
#    # image_input =
#    # Note: x should be the same name as your network's tensorflow data placeholder variable
#    # If you get an error tf_activation is not defined it may be having trouble
#    #accessing the variable from inside a function
#    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
#    featuremaps = activation.shape[3]
#    plt.figure(plt_num, figsize=(15,15))
#    for featuremap in range(featuremaps):
#        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
#        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
#        if activation_min != -1 & activation_max != -1:
#            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
#        elif activation_max != -1:
#            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
#        elif activation_min !=-1:
#            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
#        else:
#            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
#            
#            
#            
#            
#test1=X_train[6500]
#plt.imshow(test1)
#test1= (test1- np.mean(test1)) / 255.0
#outputFeatureMap(test1)