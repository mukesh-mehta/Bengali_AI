from torchvision import models
from torch import nn

pretrained_models = {'resnet18': models.resnet18,
                     'resnet34': models.resnet34,
                     'resnet50': models.resnet50,
                     'resnet101': models.resnet101,
                     'resnet152': models.resnet152,
                     'resnext50_32x4d': models.resnext50_32x4d,
                     'resnext101_32x8d': models.resnext101_32x8d}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    print("Loading {} model.".format(model_name))
    model_ft = None
    model_ft = pretrained_models[model_name](pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

if __name__ == '__main__':
    # Initialize the model for this run
    model = initialize_model("resnet18", 10, "True", use_pretrained=True)
    # Print the model we just instantiated
    print(model)
