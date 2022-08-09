import tensorflow as tf
from losses import style_loss, content_loss, total_loss

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def optim_step(generated_image, cv_model_outputs, style_layers, content_image_outputs, style_image_outputs, alpha, beta):
    with tf.GradientTape() as tape:
        # on utilise les représentations calculées précédemment F_S et F_C
        # F_G est calculée comme la sortie de vgg_model_outputs de l'image générée
        

        generated_image_outputs = cv_model_outputs(generated_image)
        
        # calculer le cout style

        L_style = style_loss(style_image_outputs, generated_image_outputs, style_layers)


        # calculer le cout contenu
        L_content = content_loss(content_image_outputs, generated_image_outputs)


        # calculer le cout total
        L = total_loss(L_content, L_style, alpha = alpha, beta = beta)
        
        
    grad = tape.gradient(L, generated_image)

    OPTIMIZER.apply_gradients([(grad, generated_image)])

    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))

    return L