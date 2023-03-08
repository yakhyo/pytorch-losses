# Common loss functions using PyTorch

## Table of Contents

- [Binary Cross-Entropy Loss](#) - ❎
- [Categorical Cross-Entropy Loss](#) - ❎
- [Mean Squared Error Loss](#) - ❎
- [Mean Absolute Error Loss](#) - ❎
- [Focal Loss](./losses/focal_loss.py) - ✅
- [Dice Loss](./losses/dice_loss.py) and [Dice Cross Entropy Loss](./losses/dice_ce_loss.py) - ✅
- [Poly Loss](./losses/poly_loss.py) - ✅
- [Asymmetric Loss](./losses/asymmetric_loss.py:10), [Optimized Asymmetric Loss](./losses/asymmetric_loss.py:63) - ✅
  and [Asymmetric Single Label Loss](./losses/asymmetric_loss.py:126) - ✅


1. **Binary Cross-Entropy Loss**: This is a loss function used for binary classification tasks. It measures the
   difference
   between the predicted probability distribution and the true probability distribution. The formula for binary
   cross-entropy loss is:

   `L = -[y*log(p) + (1-y)*log(1-p)]`

   where y is the true label (0 or 1), p is the predicted probability (between 0 and 1).

2. **Categorical Cross-Entropy Loss**: This is a loss function used for multi-class classification tasks. It measures
   the
   difference between the predicted probability distribution and the true probability distribution. The formula for
   categorical cross-entropy loss is:

   `L = -sum(y*log(p))`

   where y is a one-hot encoded vector of the true label, and p is the predicted probability distribution over all the
   classes.

3. **Mean Squared Error (MSE) Loss**: This is a loss function used for regression tasks. It measures the average squared
   difference between the predicted and true values. The formula for MSE loss is:

   `L = 1/n * sum((y - p)^2)`

   where y is the true value, p is the predicted value, and n is the number of samples.

4. **Mean Absolute Error (MAE) Loss**: This is a loss function used for regression tasks. It measures the average
   absolute
   difference between the predicted and true values. The formula for MAE loss is:

   `L = 1/n * sum(|y - p|)`

   where y is the true value, p is the predicted value, and n is the number of samples.

5. **Focal Loss**: This is a loss function designed to address class imbalance problems in classification tasks. It
   assigns
   higher weights to hard examples (i.e., those that are misclassified with high confidence). The formula for focal loss
   is:
   `L = -alpha * (1-p)^gamma * log(p)`

   where p is the predicted probability, alpha is a balancing factor between positive and negative examples, and gamma
   is a
   focusing parameter that controls the weight assigned to hard examples.

6. **Dice Loss**: This is a loss function used for segmentation tasks. It measures the overlap between the predicted and
   true segmentation masks. The formula for dice loss is:

   `L = 1 - (2*|X intersect Y| + smooth) / (|X| + |Y| + smooth)`

   where X is the predicted mask, Y is the true mask, |.| denotes the number of pixels, and smooth is a smoothing factor
   to
   avoid division by zero.
