import os

from flask import Flask, request
import numpy as np
import tensorflow as tf
import json
from flask_socketio import SocketIO,emit
from PIL import Image
from werkzeug.utils import secure_filename
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
LABEL_MAP_NAME = 'label_map.pbtxt'

foodname = ""

paths = {
    # 'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'ANNOTATION_PATH': os.path.join('tensor'),
    # 'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'CHECKPOINT_PATH': os.path.join('tensor'),
}
files = {
    'PIPELINE_CONFIG': os.path.join('tensor', 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-6')).expect_partial()

app = Flask(__name__)
socketio = SocketIO(app)


def load_labels():
    with open("label.txt", 'r') as f:
        return [line.strip() for line in f.readlines()]


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


@app.route('/', methods=['GET'])
def home():
    return 'hi'


@app.route('/img', methods=['GET', 'POST'])
@socketio.on('send_food')
def upload_file():
    if request.method == 'GET' or request.method == 'POST':
        f = request.files['imageFile']

        f.save('/img/' + secure_filename(f.filename))

        im = Image.open(f)

        im = im.resize((300, 300))

        im = np.array(im)

        input_tensor = tf.convert_to_tensor(np.expand_dims(im, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = im.copy()

        dict1 = dict(zip(detections['detection_scores'], (detections['detection_classes'] + label_id_offset)))

        category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

        before = 0

        for i in list(dict1.keys()):
            if i > 0.3:
                print(category_index.get(dict1.get(i)), ':', i * 100)

                if i > before:
                    global foodname
                    foodname = str(category_index.get(dict1.get(i)).get('name'))
                    print('foodname:::', foodname)

                before = i

        ini_string = {'foodname' : foodname}

        ini_string = json.dumps(ini_string)
        final_dictionary = json.loads(ini_string)

        socketio.emit('result', final_dictionary)
        

    return final_dictionary



   
    
    

   



if __name__ == '__main__':
   socketio.run(app, host="0.0.0.0", debug=True, port=5000)
