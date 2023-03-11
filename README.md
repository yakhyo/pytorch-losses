# Project Title

This project implements different loss functions using PyTorch.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Loss Functions](#loss-functions)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

## Installation

## Usage

```python
from losses import CrossEntropyLoss, DiceLoss, DiceCELoss, FocalLoss

# input is logits of (N, *)
# target is (N, C), C is class index
criterion = DiceLoss()
loss = criterion(input, target)
```

## Loss Functions

This project implements the following loss functions:

| Loss Name                      | Status          | Link                                                    | Task         |
|--------------------------------|-----------------|---------------------------------------------------------|--------------|
| Cross-entropy loss             | ✅ passed        | [cross_entropy_loss.py](./losses/cross_entropy_loss.py) | Segmentation |
| Binary cross-entropy loss      | Row 2, Column 2 | Row 2, Column 3                                         |              |
| Mean squared error (MSE) loss  | Row 3, Column 2 | Row 3, Column 3                                         |              |
| Mean absolute error (MAE) loss | Row 4, Column 2 | Row 4, Column 3                                         |              |
| Dice loss                      | ✅ passed        | [dice_loss.py](./losses/dice_loss.py)                   | Segmentation |
| Dice Cross Entropy loss        | ✅ passed        | [dice_loss.py](./losses/dice_loss.py)                   | Segmentation |
| Focal loss                     | ✅ passed        | [focal_loss.py](./losses/focal_loss.py)                 | Segmentation |

## Examples

[//]: # (We have included several examples in the `examples` directory to demonstrate the usage of the implemented loss)

[//]: # (functions. Each example includes a README file with detailed instructions on how to run it.)

## Contributing

Contributions are welcome! If you have a suggestion for a new loss function or an improvement to an existing one, please
open an issue or a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
