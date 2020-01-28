import os
import numpy as np
from .utils import load_image
from torch.utils import data


class DataLoader(data.Dataset):
    def __init__(self, IMG_DIR, file_list, is_test=False, transform=None):
        self.is_test = is_test
        self.root_path = IMG_DIR
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]
        image_path = os.path.join(self.root_path, file_id[0])
        image = load_image(image_path)
        if self.transform:
            image = self.transform(image)
        if self.is_test is True:
            return image
        else:
            return image, file_id[1], file_id[2], file_id[3]
