import tensorflow as tf


def load_and_prep(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, (224, 224))

    return img


class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def pred_model(filepath):
    img = load_and_prep(filepath)
    model = tf.keras.models.load_model(".\BrainTumorIdentificationModel")

    with tf.device('/cpu:0'):
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]

    return pred_class, pred_prob.max()





# @app.route('/brain-success')
# def brainS():
#     model = tf.keras.load_model('BrainTumorIdentificationModel')
#
#     target_img = os.path.join(os.getcwd(), 'static/images')
#     if request.method == 'POST':
#         if request.files:
#             file = request.files['file']
#             model_name = request.form.get('models')
#             if file and allowed_file(file.filename):
#                 file.save(os.path.join(target_img, file.filename))
#                 img_path = os.path.join(target_img, file.filename)
#                 img = file.filename
#
#                 class_result, prob_result = pred_model(img_path)
#
#                 # predictions = (class_result , int(prob_result*100))
#                 predictions = (class_result, prob_result)
#
#             return render_template("success.html", img=img, predictions=predictions, name=model_name)
#         else:
#             return render_template("index.html")