#%%
# ライブラリをインポートします
from __future__ import print_function, division
import dlib
import matplotlib.pyplot as plt
import scipy
import keras.backend as K
import PIL
from keras.utils import to_categorical
from keras import losses
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise, add
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.datasets import mnist
from imutils import face_utils
import cv2
import numpy as np

#%%
# 画像処理用のクラスを作成します
class FaceEditor:
    mask_idx = [3, 30, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3]

    def __init__(self):
        self.face_detactor = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat")

    def face_extraction(self, img):
        # 入力画像から顔を検出し配列で返す
        faces = self.face_detactor(img)
        out = []
        for face in faces:
            a = img[face.top():face.bottom()+1, face.left():face.right()+1]
            if(min(a.shape) == 0):
                continue
            out.append(img[face.top():face.bottom() +
                       1, face.left():face.right()+1])
        return out

    def clip_mouth(self, img):
        # 顔から、口部分を塗りつぶし、塗りつぶした部分を1にした配列を返す
        landmark = self.face_predictor(
            img, dlib.rectangle(0, 0, img.shape[0], img.shape[1]))
        landmark = face_utils.shape_to_np(landmark)
        mask_points = np.array(landmark[self.mask_idx]).reshape(
            (-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(img, [mask_points], (0, 0, 0))

        template = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
        template = cv2.fillPoly(template, [mask_points], (1,1,1))
        return template

    def combine(self, generated_img, original_img, template):
        # 口部分を保管した画像の口の部分だけを元の画像に合成する
        for i in range(template.shape[0]):
            for j in range(template.shape[1]):
                if(template[i][j] == 0):
                    generated_img[i][j][0] = original_img[i][j][0]
                    generated_img[i][j][1] = original_img[i][j][1]
                    generated_img[i][j][2] = original_img[i][j][2]

    def show(self, img):
        cv2.imshow('tmp', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, name,img):
        cv2.imwrite(name, img)

#%%
# 画像を読み込み、前処理を行います
match = None
helper = FaceEditor()
def load_data():
    import glob
    global train_original, train_masked, train_template, img_rows, img_cols, match
    img_rows = 128
    img_cols = 128
    match = glob.glob("C:/Users/aaa/Documents/lfw/lfw/*.jpg")
    train_original = []
    train_masked = []
    train_template = []
    for path in match:
        raw_img = cv2.imread(path)
        for original in helper.face_extraction(raw_img):
            if(min(original.shape) == 0):
                helper.show(raw_img)
                print(path)
            a = np.array(PIL.Image.fromarray(
                original).resize((img_rows, img_cols)))
            b = np.copy(a)
            train_original.append(a)
            train_template.append(helper.clip_mouth(b))
            train_masked.append(b)
    train_original = np.array(train_original) / 255
    train_masked = np.array(train_masked) / 255
    train_template = np.array(train_template) 
load_data()


#%%
# モデルを生成します
def build_generator():
    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                   padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    masked_img = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(masked_img , gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*16)

    # Upsampling
    u0 = deconv2d(d5, d4, gf*8)
    u1 = deconv2d(u0, d3, gf*4)
    u2 = deconv2d(u1, d2, gf*2)
    u3 = deconv2d(u2, d1, gf)
    u4 = UpSampling2D(size=2)(u3)
    gen_img = Conv2D(channels, kernel_size=4, strides=1,
                        padding='same', activation='tanh')(u4)

    mask_template = Input(shape=img_shape)
    without_mask = Input(shape=img_shape)
    mouth = multiply([gen_img, mask_template])
    output_img = add([mouth, without_mask])

    return masked_img,mask_template,without_mask,output_img


def build_discriminator():

    img = Input(shape=img_shape)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2,
              padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.8))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(InstanceNormalization())
    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(InstanceNormalization())


    img = Input(shape=img_shape)
    features = model(img)

    label = Flatten()(features)
    validity = Dense(1, activation="softmax")(label)

    return Model(img, validity)

def save_model():
    def save(model, model_name):
        model_path = "saved_model/%s.json" % model_name
        weights_path = "saved_model/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                   "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, "ccgan_generator")
    save(discriminator, "ccgan_discriminator")

#

#
#    original = helper.face_extraction(raw_img)[0]
#    generated = np.copy(original)
#    template = aaa.clip_mouth(generated)
#    print(template.shape)




#%%
# 学習を開始します
img_rows = 128  # 縦
img_cols = 128  # 横
channels = 3  # 色の数
img_shape = (img_rows, img_cols, channels)

# 生成器と識別機の1stレイヤーの中のフィルター数
gf = 32
optimizer = Adam(0.0002, 0.5)
epochs = 20000
batch_size = 64
sample_interval = 10

# 識別機
discriminator = build_discriminator()
discriminator.compile(loss=['binary_crossentropy'],
                      optimizer=optimizer,
                      metrics=['accuracy'])


# 生成器
discriminator.trainable = False
masked_img, mask_template, without_mask, output_img = build_generator()

generator = Model([masked_img,mask_template,without_mask], output_img)

valid = discriminator(output_img)

combined = Model([masked_img,mask_template,without_mask], valid)
combined.compile(loss=['binary_crossentropy'],
                 optimizer=optimizer)


# Adversarial ground truths
valid = np.ones(batch_size)
fake = np.zeros(batch_size)

for epoch in range(epochs):

    idx = np.random.randint(0, len(match), batch_size)
    original_imgs = train_original[idx]
    masked_imgs = train_masked[idx]
    mask_template = train_template[idx]
    without_mask = np.logical_not(mask_template) * original_imgs

    # Train the discriminator
    gen_imgs = generator.predict([masked_imgs,mask_template,without_mask])
    d_loss_real = discriminator.train_on_batch(original_imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    g_loss = combined.train_on_batch([masked_imgs,mask_template,without_mask], valid)

    # Plot the progress
    print(epoch,"d_loss_real", d_loss_real, "d_loss_fake", d_loss_fake,"g_loss",g_loss)

    # If at save interval => save generated image samples
    if epoch % sample_interval == 0:
        # Select a random half batch of images
        # idx = np.random.randint(0, batch_size, 6)
        img = np.clip(gen_imgs[0] * 255, 0, 255).astype(np.uint8)
        helper.save(str(epoch) + ".png",img)
        save_model()

