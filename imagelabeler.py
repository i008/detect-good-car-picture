from utils import list_images

from IPython.display import display
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets


class SimpleImageLabaler:
    def __init__(self, images_path, label_options=None):
        if label_options is None:
            label_options = ['front-side', 'front', 'side', 'back']

        self.images_path = images_path
        self.label_options = label_options
        self.all_buttons = [widgets.Button(description=s) for s in label_options]

        for b in self.all_buttons:
            b.on_click(self._label_click)

        self.previous_button = widgets.Button(description='previous')
        self.next_button = widgets.Button(description='next')
        self.previous_button.on_click(self._go_to_previous)
        self.next_button.on_click(self._go_to_next_pic)

        self.images_list_label = {i: 'noclass' for i in list(list_images(images_path))[:20]}
        self.slider = widgets.IntSlider(min=0, max=len(self.images_list_label))

    def _go_to_previous(self, b):
        self.slider.value -= 1

    def _go_to_next_pic(self, b):
        self.slider.value += 1

    def _label_click(self, b):
        self.images_list_label[self.current_image] = b.description
        self.slider.value += 1

    def _show_images(self, image_id):
        im = list(self.images_list_label.keys())[image_id]
        self.current_image = im
        plt.gcf()
        plt.imshow(plt.imread(im))
        name = im.split('/')[-1]
        plt.title(name + ' class: ' + self.images_list_label[im])

    def start(self):
        self.interact = interact(self._show_images, image_id=self.slider)
        display(widgets.HBox((self.previous_button, self.next_button)))
        display(*self.all_buttons)

    def save(self):
        pass
