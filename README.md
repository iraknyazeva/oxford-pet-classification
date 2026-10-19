# Oxford-IIIT Pet Classification

Многоклассовая классификация 37 пород кошек и собак (Oxford-IIIT Pet)

# Задача
- Датасет: `torchvision.datasets.OxfordIIITPet` (37 классов, trainval/test).
- Метрики: accuracy + macro f1
- Эксперименты: обучение с нуля и дообучение (голова / голова + последние слои) для ResNet18, GoogLeNet, EfficientNet-B0.

# Результаты 
## Обучение с нуля: 
  ResNet18(`best_resnet18_scratch.pth`)
      accuracy : 0.6168
            f1 : 0.6142
  GoogleNet(`best_googlenet_scratch.pth` )
      accuracy : 0.5179
            f1 : 0.5078
  EfficintNet-B0( `best_efficientnet_b0_scratch.pth`)
      accuracy : 0.6877
            f1 : 0.6832

## Дообучение головы:
  ResNet18(`best_resnet18_finetune_head.pth` )
      accuracy : 0.8763
            f1 : 0.8751
  GoogleNet(`best_googlenet_finetune_head.pth`)
      accuracy : 0.7978
            f1 : 0.7960
  EfficientNet-B0(`best_efficientnet_b0_finetune_head.pth`)
      accuracy : 0.8863
            f1 : 0.8847

## Дообучение с разморозкой последних слоев:
  ResNet18(`best_resnet18_finetune_layer4.pth`) - разморозка layer4
      accuracy : 0.8831
            f1 : 0.8811
  EfficientNet-B0(`best_efficientnet_b0_finetune_last.pth`) - разморозка последних блоков
      accuracy : 0.9032
            f1 : 0.9017


# Выводы

- Обучение с нуля в целом приемлемое качество дало для обучения на небольшом количестве данных. Лучший результат показала EfficientNet-B0. Но обучение было долгим и сходимость не очень хорошая
- Дообучение с разморозкой головы показало результат заметно лучше, лучший результат также у EfficientNet-B0, но ResNet18 тоже хороший результат показала
- На моделях, показавших более хорошее качество, попробовала разморозить и дообучить последние слои. Для этого брала модели с дообученной головой с прошлого шага. 
    - у ResNet18 размораживала layer4
    - у EfficientNet-B0 последние блоки[5:]
- Разморозка последних слоев дала небольшой прирост качества. 

В итоге лучший результат у EfficientNet-B0 - accuracy : 0.9032

# Структура репозитория

├── configs/                 # YAML-конфиги всех экспериментов
├── scripts/
│   ├── train.py             # обучение
│   └── evaluate.py          # оценка на test
├── src/
│   ├── data.py              # даталоадеры + трансформации
│   ├── models.py            # создание моделей 
│   └── train_utils.py       # train/val loop 
├── notebooks/               # исходный ноутбук 
├── requirements.txt
├── .gitignore
└── README.md

## Запуск обучения

```bash
python scripts/train.py --config configs/resnet18_scratch.yaml
python scripts/train.py --config configs/efficientnet_b0_finetune_last.yaml
```

## Оценка обученных моделей

Скачай нужный `.pth` с Google Drive и запусти:

```bash
python scripts/evaluate.py \
  --model-name efficientnet_b0 \
  --weights /path/to/best_efficientnet_b0_finetune_last.pth
```
