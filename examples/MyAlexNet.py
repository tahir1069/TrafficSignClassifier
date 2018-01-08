import tensorflow as tf
from tensorflow.contrib.layers import flatten
# Hyperparameters
def conv_layer(x,filter_size,shape_in,shape_out,stride,name,padding,mu=0,sigma=0.1):
    conv_W = tf.Variable(tf.truncated_normal(shape=(filter_size,filter_size,
                                             shape_in,shape_out), mean = mu, stddev = sigma))
    conv_b = tf.Variable(tf.zeros(shape_out))
    conv_res   = tf.nn.conv2d(x, conv_W, strides=[1, stride, stride, 1],
                              padding=padding,name = name,use_cudnn_on_gpu=True) + conv_b
    print(name,conv_res.get_shape())     
    return conv_res
def relu_layer(conv):
    return tf.nn.relu(conv)
def pooling_layer(x, filter_size,stride,padding,name):
    pooling=tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride,
                          stride, 1], padding=padding,name = name)
    print(name ,pooling.get_shape())
    return pooling
def lrn_layer(x,name):
    return tf.nn.local_response_normalization(x, depth_radius = 2, alpha = 2e-05, beta = 0.75, bias = 1.0,name=name)
def dropout_layer(x,drop_prob=0.5):
    dropout = tf.nn.dropout(x, drop_prob)
    print("DropOut Shape:",dropout.get_shape())
    return dropout
def fc_layer(layer,shape_in,shape_out,name,mean=0,stddev=0.1):
    fc_W  = tf.Variable(tf.truncated_normal(shape=(shape_in,shape_out), mean=mean,stddev=stddev))
    fc_b  = tf.Variable(tf.zeros(shape_out))
    logits =  tf.add(tf.matmul(layer, fc_W), fc_b,name = name)
    print(name,logits.get_shape())
    return logits
def AlexNet(x):
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 16x16x32. Valid Padding.
    # TODO: Pooling. Input = 16x16x32. Output = 8x8x32. Valid Padding.
    conv = conv_layer(x,7,3,32,2,padding='VALID',name='Layer1Convolution')
    x=relu_layer(conv)
    x = pooling_layer(x,3,2,padding='VALID',name='Layer1MaxPooling')
    layer1=lrn_layer(x,name='Layer1LRN')
    # TODO: Layer 2: Convolutional. Input = 8x8x32. Output = 8x8x128.
    # TODO: Pooling. Input = 8x8x128. Output = 4X4X128j.
    conv = conv_layer(layer1,5,32,128,1,padding='SAME',name='Layer2Convolution')
    x=relu_layer(conv)
    x = pooling_layer(x,3,2,padding='VALID',name='Layer2MaxPooling')
    layer2=lrn_layer(x,name='Layer2LRN')
    # TODO: Layer 3: Convolutional. Input = 4X4X128. Output = 4X4X256.
    #Same Padding
    conv = conv_layer(layer2,3,128,256,1,padding='SAME',name='Layer3Convolution')
    layer3=relu_layer(conv)
    # TODO: Layer 4: Convolutional. Input = 4X4X256. Output = 4X4X256.
    #Same Padding
    conv = conv_layer(layer3,3,256,256,1,padding='SAME',name='Layer4Convolution')
    layer4=relu_layer(conv)
    # TODO: Layer 5: Convolutional. Input = 4X4X256. Output = 4x4x128.
    # TODO: Pooling. Input = 4x4x128. Output = 4x4x128.
    conv = conv_layer(layer4,3,256,128,1,padding='SAME',name='Layer5Convolution')
    x=relu_layer(conv)
    layer5 = pooling_layer(x,3,2,padding='VALID',name='Layer5MaxPooling')
    # TODO: Layer 6: Convolutional. Input = 4x4x128. Output = 512.
    # TODO: Flattening. Input = 2x2x128. Output = 512.
    # TODO: Dropout. 
    flattened = flatten(layer5)
    print("layer 6 flattened shape:",flattened.get_shape())
    layer6=fc_layer(flattened,512,512,name='FullConnectionLayer6')
    layer6=relu_layer(layer6)
    layer6=dropout_layer(layer6)
    # TODO: Layer 7: Convolutional. Input = 512. Output = 512.
    # TODO: Dropout. 
    layer7=fc_layer(layer6,512,512,name='FullConnectionLayer7')
    layer7=relu_layer(layer7)
    layer7=dropout_layer(layer7)
    # TODO: Layer 8: Convolutional. Input = 512. Output = 43.
    return fc_layer(layer7,512,43,name='FullConnectionLayer8')
#x=tf.placeholder(tf.float32, (None, 51,51, 3))
#logits=AlexNet(x)
