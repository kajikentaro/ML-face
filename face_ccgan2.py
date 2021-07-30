#%%
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
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.datasets import mnist
from imutils import face_utils
import cv2
import numpy as np

#%%
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

        template = np.zeros((img.shape[0], img.shape[1]))
        template = cv2.fillPoly(template, [mask_points], 1)
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

    def save(self, img):
        cv2.imwrite('tmp.png', img)

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

    img = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(img, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)

    print(img)
    print(d1)
    print(d2)
    print(d3)
    print(d4)

    # Upsampling
    u1 = deconv2d(d4, d3, gf*4)
    u2 = deconv2d(u1, d2, gf*2)
    u3 = deconv2d(u2, d1, gf)
    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(channels, kernel_size=4, strides=1,
                        padding='same', activation='tanh')(u4)
    return Model(img, output_img)


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


def sample_images(epoch, imgs):
    r, c = 3, 6

    masked_imgs = mask_randomly(imgs)
    gen_imgs = generator.predict(masked_imgs)

    imgs = (imgs + 1.0) * 0.5
    masked_imgs = (masked_imgs + 1.0) * 0.5
    gen_imgs = (gen_imgs + 1.0) * 0.5

    gen_imgs = np.where(gen_imgs < 0, 0, gen_imgs)

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        axs[0, i].imshow(imgs[i, :, :, 0], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(masked_imgs[i, :, :, 0], cmap='gray')
        axs[1, i].axis('off')
        axs[2, i].imshow(gen_imgs[i, :, :, 0], cmap='gray')
        axs[2, i].axis('off')
    fig.savefig("images/%d.png" % epoch)
    plt.close()


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


helper = FaceEditor()
img_rows = 128  # 縦
img_cols = 128  # 横
match = None


#%%
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
    train_original = np.array(train_original)
    train_masked = np.array(train_masked)
    train_template = np.array(train_template)


load_data()

#%%
channels = 3  # 色の数
img_shape = (img_rows, img_cols, channels)

# 生成器と識別機の1stレイヤーの中のフィルター数
gf = 32
df = 32
optimizer = Adam(0.0002, 0.5)
epochs = 20000
batch_size = 32
sample_interval = 200

# 識別機
discriminator = build_discriminator()
discriminator.compile(loss=['binary_crossentropy'],
                      optimizer=optimizer,
                      metrics=['accuracy'])


# 生成器
masked_img = Input(shape=img_shape)

generator = build_generator()
gen_img = generator(masked_img)

discriminator.trainable = False
valid = discriminator(gen_img)

combined = Model(masked_img, valid)
combined.compile(loss=['binary_crossentropy'],
                 optimizer=optimizer)

# Adversarial ground truths
valid = np.ones(batch_size)
fake = np.zeros(batch_size)

for epoch in range(epochs):

    idx = np.random.randint(0, len(match), batch_size)
    original_imgs = train_original[idx]
    masked_imgs = train_masked[idx]
    templates = train_template[idx]

    # Train the discriminator
    gen_imgs = generator.predict(masked_imgs)
    d_loss_real = discriminator.train_on_batch(original_imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    g_loss = combined.train_on_batch(masked_imgs, valid)

    # Plot the progress
    print("%d [D loss: %f, op_acc: %.2f%%] [G loss: %f]" %
          (epoch, d_loss[0], 100*d_loss[1], g_loss))

    # If at save interval => save generated image samples
    if epoch % sample_interval == 0 and False:
        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], 6)
        imgs = X_train[idx]
        sample_images(epoch, imgs)
        save_model()


def test():
    global test_text
    test_text = "hello"


test()
print(test_text)

# %%
