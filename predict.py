import begin
from current_models import label_encoder, model_nn
from settings import IMAGE_ROWS, IMAGE_COLS
from utils import load_image_keras


@begin.start
def run(image_path=''):
    im = load_image_keras(
        image_path,
        target_size=(IMAGE_ROWS, IMAGE_COLS)
    )

    predict_class = model_nn.predict_classes(im)
    print(model_nn.predict_proba(im))
    print(label_encoder.inverse_transform(predict_class))



