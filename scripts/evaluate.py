import sys
from pathlib import Path

# корень репозитория в sys.path, чтобы можно было импортировать модули из src
sys.path.append(str(Path(__file__).resolve().parent.parent))

import click
import torch
from sklearn.metrics import classification_report, f1_score

from src.models import get_model
from src.data import get_dataloaders


@click.command()
@click.option('--model-name', required=True, type=click.Choice(['resnet18', 'googlenet', 'efficientnet_b0']))
@click.option('--weights', required=True, type=click.Path(exists=True), help='Path to .pth checkpoint')
@click.option('--data-root', default='./data')
@click.option('--batch-size', default=32)
@click.option('--num-workers', default=0)
def main(model_name, weights, data_root, batch_size, num_workers):
    # определяем устройство (использовать GPU если доступно)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # берем только необходимое для test 
    _, _, test_loader, num_classes, class_names = get_dataloaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f'Test samples: {len(test_loader.dataset)}')

    # создаем модель нужной архитектуры
    model = get_model(model_name, num_classes, pretrained=False)

    # загружаем сохраненные веса 
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # собираем предсказания и истинные метки
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # метрики
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f'\nTest Accuracy: {acc:.4f}')
    print(f'Test Macro F1: {f1:.4f}')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))


if __name__ == '__main__':
    main()