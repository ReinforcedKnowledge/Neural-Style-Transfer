import tensorflow as tf

def content_loss(content_image_outputs, generated_image_outputs):
    a_C = content_image_outputs[-1]
    a_G = generated_image_outputs[-1]

    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_C_matrix = tf.reshape(a_C, [n_H*n_W, n_C])
    a_G_matrix = tf.reshape(a_G, [n_H*n_W, n_C])

    L_content = 1/2*tf.reduce_sum(tf.square(tf.subtract(a_C_matrix, a_G_matrix)))
    return L_content

def gram_matrix(A):
    gram = tf.matmul(A, tf.transpose(A))
    return gram

def one_style_layer_loss(a_S, a_G):
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # changer la dimension (n_H * n_W, n_C) pour avoir (n_C, n_H * n_W)
    a_S = tf.transpose(tf.reshape(a_S, [n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H*n_W, n_C]))

    # calculer les matrices de Gram
    g_S = gram_matrix(a_S)
    g_G = gram_matrix(a_G)

    # calculer le coût
    L_style_layer = 1 / (4 * n_C ** 2 * (n_H * n_W) ** 2) * tf.math.reduce_sum(tf.square(tf.subtract(g_S, g_G)))
    return L_style_layer

def style_loss(style_image_outputs, generated_image_outputs, style_layers):
    L_style = 0

    a_S = style_image_outputs[:-1]
    a_G = generated_image_outputs[:-1]

    for i, weight in zip(range(len(a_S)), style_layers):  
        L_style_layer = one_style_layer_loss(a_S[i], a_G[i])

        # pondérer par le poids et ajouter au cout style total
        L_style += weight[1] * L_style_layer

    return L_style

@tf.function()
def total_loss(L_content, L_style, alpha, beta):
    L = alpha * L_content + beta * L_style
    return L