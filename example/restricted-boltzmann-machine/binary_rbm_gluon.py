# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import random as pyrnd
import numpy as np
import mxnet as mx
from matplotlib import pyplot as plt
from binary_rbm import BinaryRBMBlock

mx.random.seed(pyrnd.getrandbits(32))
ctx = mx.gpu()

### Set hyperparameters

num_hidden = 500
k = 20 # PCD-k
batch_size = 10
num_epoch = 10
learning_rate = 0.1 # The optimizer rescales this with `1 / batch_size`


### Prepare data

def data_transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)

mnist_dataset = mx.gluon.data.vision.MNIST(train=True, transform=data_transform)
img_height = mnist_dataset[0][0].shape[0]
img_width = mnist_dataset[0][0].shape[1]
num_visible = img_width * img_height

# Generate arrays with shape (batch_size, height = 28, width = 28, num_channel = 1)
train_data = mx.gluon.data.DataLoader(mnist_dataset, batch_size, shuffle=True)


### Train

rbm = BinaryRBMBlock(num_hidden=num_hidden, k=k, for_training=True)
rbm.initialize(mx.init.Normal(sigma=.01), ctx=ctx)
rbm.hybridize()
trainer = mx.gluon.Trainer(
    rbm.collect_params('^(?!.*_aux_.*).*$'), # Optimize all parameters except aux states
    'sgd', {'learning_rate': learning_rate})
for e in range(num_epoch):
    for i, (batch, _) in enumerate(train_data):
        batch = batch.as_in_context(ctx).reshape((batch_size, num_visible))
        with mx.autograd.record():
            out = rbm(batch)
        out[0].backward()
        trainer.step(batch.shape[0])
    mx.nd.waitall() # To restrict memory usage
    print("Epoch %s complete" % (e,))


### Show some samples. Each sample is obtained by 1000 steps of Gibbs sampling starting from a real data.

showcase_gibbs_sampling_steps = 1000
showcase_num_samples_w = 15
showcase_num_samples_h = 15
showcase_num_samples = showcase_num_samples_w * showcase_num_samples_h
showcase_img_shape = (showcase_num_samples_h * img_height, showcase_num_samples_w * img_width)
showcase_img_column_shape = (showcase_num_samples_h * img_height, img_width)

showcase_rbm = BinaryRBMBlock(
    num_hidden=num_hidden,
    k=showcase_gibbs_sampling_steps,
    for_training=False,
    params=rbm.collect_params('^(?!.*_aux_.*).*$'))
showcase_iter = iter(mx.gluon.data.DataLoader(mnist_dataset, showcase_num_samples_h, shuffle=True))
showcase_img = np.zeros(showcase_img_shape)
for i in range(showcase_num_samples_w):
    data_batch = next(showcase_iter)[0].as_in_context(ctx).reshape((showcase_num_samples_h, num_visible))
    sample_batch = showcase_rbm(data_batch)
    # Each pixel is the probability that the unit is 1.
    showcase_img[:, i * img_width : (i + 1) * img_width] = sample_batch[0].reshape(showcase_img_column_shape).asnumpy()
s = plt.imshow(showcase_img, cmap='gray')
plt.axis('off')
plt.show(s)
