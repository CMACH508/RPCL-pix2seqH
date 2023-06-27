#   Copyright 2023 Sicong Zang
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
""" Latent space visualization """

from tsne_clustering import arrangeTsneByDist, saveImage2Path, sort_paths, get_data, applyTSNE, get_data
import glob
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import utils
from model import Model
import tensorflow as tf
import json
plt.switch_backend('agg')

def load_model_params(model_dir):
    model_params = utils.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_config = json.dumps(json.load(f))
        model_params.parse_json(model_config)
    return model_params

model_params = load_model_params('model')
code_length = model_params.z_size
category = len(model_params.categories)
color = ['black', 'red', 'blue', 'green', 'orange', 'cyan', 'tomato', 'magenta', 'purple', 'brown',
         'teal', 'peru', 'darkviolet', 'gold', 'pink', 'lime', 'olive']

def main():
    for label in range(category):
        img_paths = glob.glob('./sample/%d_*.png' % label)  # Directory for generations
        code_paths = glob.glob('./sample/code_%d_*.npy' % label)  # Directory for latent codes
        i_label_paths = glob.glob('./sample/i_label_%d_*.npy' % label)  # Directory for posteriors
        j_label_paths = glob.glob('./sample/j_label_%d_*.npy' % label)  # Directory for posteriors
        img_paths = sort_paths(img_paths)
        code_paths = sort_paths(code_paths)
        i_label_paths = sort_paths(i_label_paths)
        j_label_paths = sort_paths(j_label_paths)
        if label == 0:
            img = np.array(img_paths)
            code = np.array(code_paths)
            pre_i_label = np.array(i_label_paths)
            pre_j_label = np.array(j_label_paths)
        else:
            img = np.hstack((img, np.array(img_paths)))
            code = np.hstack((code, np.array(code_paths)))
            pre_i_label = np.hstack((pre_i_label, np.array(i_label_paths)))
            pre_j_label = np.hstack((pre_j_label, np.array(j_label_paths)))

    img_data = []
    for path in img:
        img_data.append(Image.open(path).convert(mode='RGB'))
    code_data = []
    for path in code:
        code_data.append(np.load(path))
    code_data = np.reshape(code_data, [-1, code_length])
    i_label_data = []
    for path in pre_i_label:
        i_label_data.append(np.load(path))
    i_label_data = np.array(i_label_data, dtype=np.int32)
    i_label_data = np.reshape(i_label_data, [-1])
    j_label_data = []
    for path in pre_j_label:
        j_label_data.append(np.load(path))
    j_label_data = np.array(j_label_data, dtype=np.int32)
    j_label_data = np.reshape(j_label_data, [-1])

    model_params = load_model_params("./model/")
    model_params.batch_size = 1
    model = Model(model_params)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())
    utils.load_checkpoint(sess, "./model/")
    mu = sess.run(model.de_mu)

    # T-sne clustering
    tsne = applyTSNE(np.concatenate([code_data, mu], axis=0))
    tsne_mu = tsne[len(code_data):len(tsne)]
    tsne = tsne[0:len(code_data)]
    grid_image = arrangeTsneByDist(tsne, img_data, width=2000, height=1500, max_dim=48)
    saveImage2Path(grid_image, './result.jpg')

    plt.figure(1)
    for kk in range(model_params.num_layer_j):
        if kk == 0:
            marker = str(kk)
        else:
            marker = np.hstack((marker, str(kk)))
    marker = list(marker)
    idx = 0
    for k in range(model_params.num_layer_j):
        temp = np.argwhere(j_label_data == k)
        temp = np.squeeze(temp, axis=1)
        if len(temp) > 1:
            plt.scatter(tsne[temp, 0], -tsne[temp, 1], marker='.', c=color[k])
            marker[idx] = "#" + str(k)
            idx += 1
    # plt.scatter(tsne_mu[:, 0], -tsne_mu[:, 1], marker='x', c='black')
    plt.legend(marker)
    plt.savefig('./j.jpg')

    plt.figure(2)
    for kk in range(model_params.num_layer_i):
        if kk == 0:
            marker = str(kk)
        else:
            marker = np.hstack((marker, str(kk)))
    marker = list(marker)
    idx = 0
    for k in range(model_params.num_layer_i):
        temp = np.argwhere(i_label_data == k)
        temp = np.squeeze(temp, axis=1)
        if len(temp) > 1:
            plt.scatter(tsne[temp, 0], -tsne[temp, 1], marker='.', c=color[k])
            marker[idx] = "#" + str(k)
            idx += 1
    plt.legend(marker)
    plt.savefig('./i.jpg')

if __name__ == '__main__':
    main()
