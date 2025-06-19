# models.py

import torchvision.models as models
from model.cnn_8layers import CNN8Layer
from model.cnn_16layers import CNN16Layer
from model.resnet18 import ResNet18
from model.cnn_2layers import CNN2Layer
from model.cnn_3layers import CNN3Layer
from model.cnn_4layers import CNN4Layer
from model.cnn_5layers import CNN5Layer
from model import resnet18
from model import resnet18k_v2
from model.cnn_5layers_custom import CNN5Layer_cus


def load_models(in_channels, args, img_size, num_classes):
    if args.model == "cnn_2layers":
        model = CNN2Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "cnn_5layers":
        model = CNN5Layer(in_channels, num_classes, args.model_width, img_size)
    elif args.model == "resnet18_lib":
        model = models.resnet18(num_classes=num_classes)
    elif args.model == "resnet18_scr":
        model = resnet18.make_resnet18(in_channels,num_classes)
        print(model)
    elif args.model == "resnet18k":
        k1 = args.model_width
        model = resnet18k_v2.make_resnet18k(k=k1, num_classes=num_classes,)
        print('make resnet18k')
    elif args.model == "resnet18_v2":
        k1 = 64 * args.model_width
        model = ResNet18_v2(num_classes=num_classes,k = k1)
        print(model)
    elif args.model == "cnn_5layers_cus":
        model = CNN5Layer_cus(in_channels, num_classes, args.model_width, img_size)
    else:
        raise ValueError("Invalid model name.")
    #print(model)
    return model

