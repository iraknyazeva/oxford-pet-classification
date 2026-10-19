import os
import time
from tqdm.auto import tqdm
import torch


# одна эпоха обучения
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train() # в режим обучения
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='train', leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # обнуляем градиенты с прошлого шага
        outputs = model(images) # прямой проход
        loss = criterion(outputs, labels) 
        loss.backward() # обратный проход, считаем градиенты
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # возвращаем средние loss и accuracy за эпоху
    return running_loss / total, correct / total


@torch.no_grad() # на валидации градиенты не считаем
def validate(model, loader, criterion, device):
    model.eval() # в режим инференса 
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='val', leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, model_name, save_dir, patience=5):
    os.makedirs(save_dir, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()

        # одна эпоха обучения и валидация
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # сохраняем метрики
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # шаг sheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        elapsed = time.time() - start
        lr = optimizer.param_groups[0]['lr']

        print(
            f'epoch {epoch:02d}/{num_epochs} | '
            f'train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
            f'val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
            f'lr: {lr:.2e} | time: {elapsed:.1f}s'
        )

        # если есть улучшение, сохраняем лучшую по accuracy на валидации модель и сбрасываем счетчик ранней остановки
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            path = os.path.join(save_dir, f'best_{model_name}.pth')
            torch.save(model.state_dict(), path)
            print(f' saved best model val_acc={best_val_acc:.4f}')
        else:
            # если улучшений не было, увеличиваем счетчик
            epochs_no_improve += 1
            print(f'  no improvement for {epochs_no_improve}/{patience}')

        # ранняя остановка
        if epochs_no_improve >= patience:
            print(f'early stopping at epoch {epoch}')
            break

    return history