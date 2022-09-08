import tensorflow as tf
from losses import style_loss, content_loss, total_loss

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def optim_step(generated_image, cv_model_outputs, style_layers, content_image_outputs, style_image_outputs, alpha, beta):
    with tf.GradientTape() as tape:
        generated_image_outputs = cv_model_outputs(generated_image)

        L_style = style_loss(style_image_outputs, generated_image_outputs, style_layers)

        L_content = content_loss(content_image_outputs, generated_image_outputs)

        L = total_loss(L_content, L_style, alpha = alpha, beta = beta)
        
    grad = tape.gradient(L, generated_image)

    OPTIMIZER.apply_gradients([(grad, generated_image)])

    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))

    return L