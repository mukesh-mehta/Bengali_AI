import numpy as np
from torch.utils import data


class DataLoader(data.Dataset):
    def __init__(self, image_data, is_test=False, transform=None):
        self.is_test = is_test
        self.image_data = image_data
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        if index not in range(0, len(self.image_data)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        image = self.image_data[index][0]
        if self.transform:
            image = self.transform(image)
        if self.is_test is True:
            return image
        else:
            return image, self.image_data[index][1:]
