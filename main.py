import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from optimize import optim_step

tf.random.set_seed(0) # For replication and because streamlit reruns the whole script

IMAGES_FOLDER = './images'
IMG_SIZE = 224
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)
]
CONTENT_LAYER = [('block5_conv4', 1)]

def image_selector(key): # Put in utilities later
    filenames = os.listdir(IMAGES_FOLDER)
    selected_filename = st.selectbox('', filenames, key=key)
    return os.path.join(IMAGES_FOLDER, selected_filename)

@st.cache
def load_model():
    vgg = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights='imagenet'
    )
    vgg.trainable = False
    return vgg

@st.cache
def model_layers_outputs(cv_model, layer_names):
    outputs = [cv_model.get_layer(layer[0]).output for layer in layer_names]
    model_outputs = tf.keras.Model([cv_model.input], outputs)
    return model_outputs

st.write(
    """
    ##### Select your content image
    """
)

content_image_path = image_selector(key='content')
#st.write('You have selected `%s`' % content_image_path)
st.image(content_image_path, caption=content_image_path.split('/')[-1])

st.write( 
    """
    ##### Select your style image
    """
)

style_image_path = image_selector(key='style')
st.image(style_image_path, caption=style_image_path.split('/')[-1])

st.write(
    """
    ##### Mixed image
    """
)

content_image = np.array(Image.open(content_image_path).resize((IMG_SIZE, IMG_SIZE)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

style_image =  np.array(Image.open(style_image_path).resize((IMG_SIZE, IMG_SIZE)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

fig, ax = plt.subplots()
ax.imshow(generated_image.numpy()[0])

st.pyplot(fig)

w1 = st.number_input('Insert first weight', value = 0.2)
w2 = st.number_input('Insert second weight', value = 0.2)
w3 = st.number_input('Insert third weight', value = 0.2)
w4 = st.number_input('Insert fourth weight', value = 0.2)
w5 = st.number_input('Insert fifth weight', value = 0.2)

STYLE_LAYERS = [
    ('block1_conv1', w1),
    ('block2_conv1', w2),
    ('block3_conv1', w3),
    ('block4_conv1', w4),
    ('block5_conv1', w5)
]
CONTENT_LAYER = [('block5_conv4', 1)]

alpha = st.number_input('Insert alpha value', value = 1)
beta = st.number_input('Insert beta value', value = 1000)
num_steps = st.number_input('Insert number of steps', value = 20)

optim_start = st.selectbox(
    'Would you like to start adding style to content?',
     ('No', 'Yes')
)

if optim_start=='Yes':
    vgg = load_model()
    vgg_model_outputs = model_layers_outputs(vgg, STYLE_LAYERS + CONTENT_LAYER)

    preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    content_image_outputs = vgg_model_outputs(preprocessed_content)
    style_image_outputs = vgg_model_outputs(preprocessed_style)
    

    generated_image = tf.Variable(generated_image)

    losses = [0]*num_steps
    for t in range(num_steps):
        L = optim_step(
            generated_image, vgg_model_outputs, STYLE_LAYERS, 
            content_image_outputs, style_image_outputs, alpha, beta
        )
        losses[t] = L

    st.image(content_image_path, caption='Original content image')

    w, h = Image.open(content_image_path).size
    mixed_image = Image.fromarray(np.array(generated_image[0] * 255, dtype=np.uint8)).resize((w, h))
    st.image(mixed_image, caption='Mixed image')

    fig, ax = plt.subplots()
    ax.plot(range(num_steps), losses)
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Total cost')
    st.pyplot(fig)

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(content_image[0])
    ax.title.set_text('Content image')
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(style_image[0])
    ax.title.set_text('Style image')
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(generated_image[0])
    ax.title.set_text('Mixed image')
    st.pyplot(fig)