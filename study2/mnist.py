import os
import struct
import numpy as np
import matplotlib.pyplot as plt

path="e:/tf/mnist/";kind="train"
lbpath=open(labels_path, 'rb')
def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


import numpy as np
import struct
import matplotlib.pyplot as plt

filename = 'train-images.idx3-ubyte'
binfile = open('e:/tf/mnist/'+filename, 'rb')
buf = binfile.read()

index = 0
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

im = struct.unpack_from('>784B', buf, index)
index += struct.calcsize('>784B')

im = np.array(im)
im = im.reshape(28, 28)

fig = plt.figure()
plotwindow = fig.add_subplot(111)
plt.imshow(im, cmap='gray')
plt.show()

magic, num, rows, cols = struct.unpack('>IIII', binfile.read(16))
images = np.fromfile(binfile, dtype=np.uint8)
images=images.reshape(-1,784)


####################################################
X_train, y_train = load_mnist("e:/tf/mnist/", kind="train")
X_test, y_test = load_mnist("MNIST_data/", kind="t10k")

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(25):
    img = X_train[y_train == 9][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()





from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import pickle
import _pickle as pickle
with open('e:/tf/mnist/data1.pkl', 'wb') as f:
    pickle.dump(mnist, f)
# 反序列化
with open('e:/tf/mnist/data.pkl', 'rb') as f:
    mnist = pickle.load(f)

import sys
sys.getsizeof(a)

import json
with open('e:/tf/mnist/data2.json', 'w', encoding='utf-8') as f:
    json.dump(mnist, f)

with open('abc.json', encoding='utf-8') as f:
    obj = json.load(f, object_hook=dict2person)
    print(obj.name, obj.age, obj.job)
    obj.work()

print('Training data size: ', mnist.train.num_examples)
print('Validation data size: ', mnist.validation.num_examples)
print('Test data size: ', mnist.test.num_examples)

img0 = mnist.train.images[0].reshape(28,28)
img1 = mnist.train.images[1].reshape(28,28)
img2 = mnist.train.images[2].reshape(28,28)
img3 = mnist.train.images[3].reshape(28,28)

a=mnist.train

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(221)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)

ax0.imshow(img0)
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(img3)
fig.show()
