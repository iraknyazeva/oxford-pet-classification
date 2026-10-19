import sys
from pathlib import Path

# корень репозитория в sys.path для импорта из src
sys.path.append(str(Path(__file__).resolve().parent.parent))

import click
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders, set_seed
from src.models import get_model, freeze_backbone
from src.train_utils import train_model


@click.command()
@click.option('--config', default='configs/default.yaml', help='Path to YAML config')
def main(config):
    # конфиг эксперимента
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # фиксируем seed и устройство
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # получаем даталоадеры для обучения и валидации
    train_loader, val_loader, _, num_classes, _ = get_dataloaders(
        root=cfg['data']['root'],
        batch_size=cfg['data']['batch_size'],
        val_ratio=cfg['data']['val_ratio'],
        num_workers=cfg['data']['num_workers'],
        seed=cfg['seed'],
    )

    # создаем модель
    model = get_model(
        name=cfg['model']['name'],
        num_classes=num_classes,
        pretrained=cfg['model']['pretrained'],
    )

    # при необходимости замораживаем все и размораживаем указанные слои
    if cfg['model']['freeze_backbone']:
        model = freeze_backbone(
            model,
            cfg['model']['name'],
            unfreeze_layers=cfg['model'].get('unfreeze_layers'),
        )

    model = model.to(device)

    # фиксируем функцию потерь, оптимизатор, sheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['lr'],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    # имя модели для дальнейшего сохранения
    mode = 'ft' if cfg['model']['pretrained'] else 'scratch'
    model_name = f'{cfg['model']['name']}_{mode}'
    if cfg['model'].get('unfreeze_layers'):
        model_name += f'_{cfg['model']['unfreeze_layers']}'

    # запускаем обучение
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=cfg['training']['epochs'],
        device=device,
        model_name=model_name,
        save_dir=cfg['training']['save_dir'],
        patience=cfg['training']['patience'],
    )

    print('Training finished.')
    print(f'Best val acc: {max(history['val_acc']):.4f}')


if __name__ == '__main__':
    main()