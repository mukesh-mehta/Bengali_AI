import numpy as np
from torch.utils import data
from torchvision import transforms


class ImageLoader(data.Dataset):
    def __init__(self, labels_df, image_df, is_test=False, transform=transforms.Compose([transforms.ToTensor()])):
        self.is_test = is_test
        self.transform = transform
        if not self.is_test:
            self.labels_df = labels_df.values
        self.image_df = image_df

    def __len__(self):
        return self.image_df.shape[0]

    def __getitem__(self, index):
        if index not in range(0, self.image_df.shape[0]):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        image = np.repeat(self.image_df.iloc[index].values[1:].reshape(-1, 137, 236).astype(int), 3, 0)
        # image = self.image_df.iloc[index].values[1:].reshape(-1, 137, 236).astype(int)
        if self.transform:
            image = self.transform(image)
        if self.is_test is True:
            return image
        else:
            labels = self.labels_df[index]
            return image, labels