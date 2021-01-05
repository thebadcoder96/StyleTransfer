import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import functools
import argparse
from tensorflow.keras.preprocessing import image as kp_image


class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def vgg_layers(self, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class Styletransfer():

    def __init__(self, args):
        self.content_image = self.load_img(args.content_path)
        self.style_image = self.load_img(args.style_path)
        self.opt = tf.optimizers.Adam(
            learning_rate=args.lr, beta_1=args.beta1, epsilon=args.epsilon)
        self.style_weight = args.style_weight
        self.content_weight = args.content_weight
        self.total_variation_weight = args.tv_weight
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.extractor = StyleContentModel(
            self.style_layers, self.content_layers)
        self.results = self.extractor(tf.constant(self.content_image))
        self.style_targets = self.extractor(self.style_image)['style']
        self.content_targets = self.extractor(self.content_image)['content']
        self.iterations = args.iterations
        self.checkpoint_iterations = args.checkpoint_iterations
        self.image = tf.Variable(self.content_image)

    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    '''def load_img(self, path_to_img):
        max_dim = 512
        img = PIL.Image.open(path_to_img)
        long = max(img.size)
        scale = max_dim/long
        img = img.resize(
            (round(img.size[0]*scale), round(img.size[1]*scale)), PIL.Image.ANTIALIAS)

        img = kp_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        return img'''

    def load_img(self, path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += self.total_variation_weight*tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        self.image.assign(self.clip_0_1(image))

    def run(self):
        epochs = 10
        steps_per_epoch = self.iterations
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step(self.image)
            if (step % self.checkpoint_iterations == 0):
                print('Iteration: {}'.format(step))  
                self.tensor_to_image(self.image).save(f"output_{step}_iter.png")

        file_name = 'stylized-image.png'
        self.tensor_to_image(self.image).save(file_name)


def build_parser():
    parser = argparse.ArgumentParser()
    ITERATIONS = 500
    CONTENT_WEIGHT = 1e4
    STYLE_WEIGHT = 1e-2
    TV_WEIGHT = 30
    BETA1 = 0.99
    LR = 0.02
    EPSILON = 1e-1
    CHECKPOINT = 100
    parser.add_argument('--content',
                        dest='content_path', help='path to content image',
                        metavar='CONTENT', required=True)
    parser.add_argument('--style',
                        dest='style_path', help='path to style image',
                        metavar='STYLE', required=True)
    parser.add_argument('--output',
                        dest='output', help='output path',
                        metavar='OUTPUT', required=True)
    parser.add_argument('--iterations', type=int,
                        dest='iterations', help='Number of iterations (default %(default)s) per EPOCH',
                        metavar='ITERATIONS', default=ITERATIONS)
    parser.add_argument('--checkpoint-iterations', type=int,
                        dest='checkpoint_iterations', help='checkpoint frequency; Save every n iterations',
                        metavar='CHECKPOINT_ITERATIONS', default=CHECKPOINT)
    parser.add_argument('--content-weight', type=float,
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT', default=CONTENT_WEIGHT)
    parser.add_argument('--style-weight', type=float,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT', default=STYLE_WEIGHT)
    parser.add_argument('--tv-weight', type=float,
                        dest='tv_weight',
                        help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT', default=TV_WEIGHT)
    parser.add_argument('--learning-rate', type=float,
                        dest='lr', help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LR)
    parser.add_argument('--beta1', type=float,
                        dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
                        metavar='beta1', default=BETA1)
    parser.add_argument('--eps', type=float,
                        dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
                        metavar='epsilon', default=EPSILON)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    Styletransfer(args).run()


if __name__ == "__main__":
    main()
