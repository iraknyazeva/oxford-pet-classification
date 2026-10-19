import torch.nn as nn
from torchvision import models



'''
создает модель из torchvision и заменяет голову под количество классов
    
name: название архитектуры ('resnet18', 'googlenet', 'efficientnet_b0')
num_classes: количество классов в задаче
pretrained: если True — загружает веса ImageNet, иначе инициализирует случайно
'''
def get_model(name: str, num_classes: int, pretrained: bool = False):

    if name == 'resnet18':
        weights = 'DEFAULT' if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == 'googlenet':
        weights = 'DEFAULT' if pretrained else None
        model = models.googlenet(weights=weights, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == 'efficientnet_b0':
        weights = 'DEFAULT' if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f'Unknown model: {name}')

    return model



'''
замораживает все слои модели и размораживает голову (+ опционально последние блоки).

model: модель, созданная через get_model
name: название архитектуры
unfreeze_layers: какие дополнительные слои разморозить
                ('layer4' для ResNet, 'features[5:]' для EfficientNet)
'''
def freeze_backbone(model, name: str, unfreeze_layers=None):
    for param in model.parameters():
        param.requires_grad = False

    if name == 'resnet18':
        for param in model.fc.parameters():
            param.requires_grad = True
        if unfreeze_layers == 'layer4':
            for param in model.layer4.parameters():
                param.requires_grad = True

    elif name == 'googlenet':
        for param in model.fc.parameters():
            param.requires_grad = True

    elif name == 'efficientnet_b0':
        for param in model.classifier[1].parameters():
            param.requires_grad = True
        if unfreeze_layers == 'features[5:]':
            for param in model.features[5:].parameters():
                param.requires_grad = True

    return model
