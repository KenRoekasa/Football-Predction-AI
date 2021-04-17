import visualkeras
import tensorflow as tf





model = tf.keras.models.load_model("D:/Desktop/kxr758/logs/regularisation/dropout/dropout2021-04-14/epoch20020210414-19500010-all-feature_importance-ratio-smote/savedmodel20210414-195852")


dot_img_file = 'tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)