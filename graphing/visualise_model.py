import visualkeras
import tensorflow as tf





model = tf.keras.models.load_model("D:/Desktop/kxr758/logs/archive/test/regularisation/2021-04-01/epoch16020210401-165916alltraining dropout/savedmodel20210401-171722")

print(model.summary())

dot_img_file = 'tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)