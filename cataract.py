import tensorflow as tf

classes = ['Cataract', 'Normal']
model = model_1
# Function to load and preprocess an image
IMG_SIZE = (224, 224)

def load_and_prep(filepath):
  img = tf.io.read_file(filepath)
  img = tf.io.decode_image(img)
  img = tf.image.resize(img, IMG_SIZE)

  return img

def pred_model(imgpath):
    img_2 = load_and_prep(imgpath)

    with tf.device('/cpu:0'):
        pred_prob = model.predict(tf.expand_dims(img_2, axis=0))
        print(pred_prob)
        pred_class = classes[pred_prob.argmax()]

    return pred_class, pred_prob.max()

img_path = "/content/drive/MyDrive/Dataseet/training/cataract/cataract_046.png"
class_result, prob_result = pred_model(img_path)
predictions = (class_result, int(prob_result * 100))

print(predictions)