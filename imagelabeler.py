import json
from collections import OrderedDict

import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, widgets

from utils import list_images


class SimpleImageLabaler:
    """
    This class implements a simple Image label widget to be used in Jupyter Notebook's
    usage:
    %matplotlib inline
    labaler = SimpleImageLabaler('path_to_images', labal_options=['a','b','c'])
    labaler.start()

    """

    def __init__(self, images_path, label_options=None, preload_labels='labels.json'):

        if not label_options:
            label_options = ['front-side', 'front', 'side', 'back', 'back-side', 'top', 'other']

        self.images_path = images_path
        self.label_options = label_options
        self.all_buttons = [widgets.Button(description=s) for s in label_options]

        self.previous_button = widgets.Button(description='previous')
        self.next_button = widgets.Button(description='next')
        self.previous_button.on_click(self._go_to_previous)
        self.next_button.on_click(self._go_to_next_pic)

        for b in self.all_buttons:
            b.on_click(self._label_click)

        if preload_labels:
            with open(preload_labels, 'r') as labels:
                self.images_list_label = json.loads(labels.read())
        else:
            self.images_list_label = {i: 'noclass' for i in list(list_images(images_path))}

        self.images_list_label = OrderedDict(sorted(self.images_list_label.items()))
        self.slider = widgets.IntSlider(min=0, max=len(self.images_list_label) - 1)
        self.slider.value = self._get_next_first_unlabeled()

    def _get_next_first_unlabeled(self):
        # numpy solution np.where(np.array(self.image_list_label.values()) == 'noclass')[0]
        return [i for i, x in enumerate(self.images_list_label.values()) if x == 'noclass'][0]

    def _go_to_previous(self, *args):
        self.slider.value -= 1

    def _go_to_next_pic(self, *args):
        self.slider.value += 1

    def _label_click(self, b):
        self.images_list_label[self.current_image] = b.description
        with open('labels.json', mode='w+') as labels:
            labels.writelines(json.dumps(self.images_list_label))
        self.slider.value = self._get_next_first_unlabeled()

    def _show_images(self, image_id):
        im = list(self.images_list_label.keys())[image_id]
        self.current_image = im
        plt.gcf()
        plt.imshow(plt.imread(im))
        plt.title(im.split('/')[-1] + ' class: ' + self.images_list_label[im])

    def start(self):
        self.interact = interact(self._show_images, image_id=self.slider)
        display(widgets.HBox((self.previous_button, self.next_button)))
        display(*self.all_buttons)
        print('[INFO] Resuming at index {}'.format(self._get_next_first_unlabeled()))

    def save(self):
        pass
