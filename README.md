# DL_CW

COMPSCI5085 Deep Learning M Coursework

## Notes

1. Design, implement and test a deep learning architecture to detect and identify damage in images.
2. Each colour in those images corresponds to a different category of damage, including fold, writing or burn marks.
3. can decide to use unsupervised pre-training of only supervised end-to-end training.
4. Dealing with a semantic segmentation problem, where the goal is to classify each pixel of an image into a predefined category.
5. Convolutional Neural Networks (CNNs) are typically the architecture of choice due to their effectiveness in image recognition and classification tasks.
6. To fit within your VRAM constraints:
   1. Start with a small batch size: Begin with a batch size that you're certain will fit in memory (e.g., 2 or 4) and increase it until you start to approach your VRAM limit.
   2. Reduce image resolution: If you're limited by VRAM, consider reducing the size of the input images. This will reduce the memory required for both the images and the feature maps.
   3. Model complexity: Use a simpler model or reduce the width (number of channels) of the U-Net layers if you're constrained by memory.
   4. Gradient accumulation: If you're unable to use a large batch size due to VRAM constraints, you can simulate larger batches by accumulating gradients over several forward/backward passes before updating the model weights.
   5. $VRAM\_usage = model\_size + (input\_size + output\_size + intermediate\_variables\_size) * batch\_size$
7. To monitor GPU usage: `nvidia-smi -l 1 --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits`

### Colour labels for each damage type

| Damage Type   | Colour Code                                     |
| ------------- | ----------------------------------------------- |
| Material loss | <span style="color:#1CE6FF">Sky Blue</span>     |
| Peel          | <span style="color:#FF34FF">Magenta</span>      |
| Dust          | <span style="color:#FF4A46">Coral Red</span>    |
| Scratch       | <span style="color:#008941">Dark Green</span>   |
| Hair          | <span style="color:#006FA6">Deep Blue</span>    |
| Dirt          | <span style="color:#A30059">Maroon</span>       |
| Fold          | <span style="color:#FFA500">Orange</span>       |
| Writing       | <span style="color:#7A4900">Brown</span>        |
| Cracks        | <span style="color:#0000A6">Blue</span>         |
| Staining      | <span style="color:#63FFAC">Mint Green</span>   |
| Stamp         | <span style="color:#004D43">Teal</span>         |
| Sticker       | <span style="color:#8FB0FF">Light Blue</span>   |
| Puncture      | <span style="color:#997D87">Taupe</span>        |
| Burn marks    | <span style="color:#809693">Slate Grey</span>   |
| Lightleak     | <span style="color:#f6ff1b">Lemon Yellow</span> |
| Background    | <span style="color:#5A0007">Dark Red</span>     |


## For report

1. The U-Net architecture has a distinctive U-shaped structure with a contracting path to capture context and a symmetric expanding path that enables precise localization.
2. Data augmentation is a preprocessing technique used to increase the diversity of your data without actually collecting new data. This is done by applying random transformations that make sense for your dataset and problem domain.
3. The VRAM needs to fit the model, the data of a single batch, and the intermediate computations that occur during the forward and backward passes.
   1. The batch size is the main factor we can control. Larger batches will require more VRAM.
   2. The complexity of the model and the size of the input images also determine VRAM usage. More complex models and larger input images require more VRAM.
4. Some image sizes are different. We need to resize them to the same size.
   1. `Input shape: torch.Size([1, 3, 1024, 1020])`
   2. This causes the division after max pooling to not be evenly divisible causing: `Shape after encoder cnn: torch.Size([1, 512, 64, 63])`
   3. Which causes more problems after flattening as `Shape after encoder flatten: torch.Size([1, 2064384])` producing error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x2064384 and 524288x2)`.

---

# Report


## Choice of Architecture

The U-Net architecture is a popular choice for semantic segmentation tasks due to its ability to capture fine-grained details, its efficient use of parameters, and its efficiency and slight dataset effectiveness. The U-Net architecture has a distinctive U-shaped structure with a down path to capture context and a symmetric expanding path that enables precise localisation. The down path is composed of a series of convolutional and max-pooling layers that gradually reduce the spatial dimensions of the input. In contrast, the upward path is composed of a series of up-convolutional and concatenation layers that gradually increase the spatial dimensions of the input. The down path captures context by learning to recognise features at different scales.
In contrast, the upward path enables precise localisation by learning to reconstruct the input from the features learned in the down path. The U-Net architecture is also known for capturing fine-grained details using skip connections to pass high-resolution features from the down path to the expanding path. This allows the U-Net architecture to capture fine-grained information lost in the down path and produce high-resolution predictions that are accurate and visually appealing.

U-Net was chosen over other architectures, such as FCN (Fully Convolutional Networks) and SegNet, because of its superior performance in tasks requiring fine detail and its unique structure that promotes better information flow between the encoding and decoding parts. FCN sometimes struggles with capturing fine details, which may result in less precise boundaries than U-Net. SegNet, on the other hand, uses a different approach to upsample and produce segmentation maps, which can be efficient but does not always provide the level of detail and accuracy that U-Net can achieve, especially in cases where the segmentation task requires high fidelity to small objects or intricate patterns.

Therefore, the U-Net architecture was chosen for this task for these reasons. Its symmetric expanding path, efficient parameter usage, effectiveness with small datasets, and the utilisation of skip connections for preserving high-resolution features throughout the network make it uniquely suited for achieving high accuracy in semantic segmentation tasks that demand precise localisation and fine-grained detail capture.

![U-Net Architecture](https://raw.githubusercontent.com/JalalSayed1/Image-Damage-Classification/ff8499ff3aea505cd5880188e1392e3a19346f82/img/UNet.ppm)

## Methodology

### Data Preprocessing

To ensure consistency in model training and testing, validation images were cropped to match the dimensions of the training set images, optimising the model's performance across uniformly sized inputs. The images were cropped the same way the training images were cropped.

### Loss Function

The model utilises cross-entropy loss, with an ignore index applied to the background class to mitigate its overwhelming presence and encourage focus on more relevant features. This loss function is a popular choice for semantic segmentation tasks as it is effective at penalising incorrect predictions and encouraging the model to learn the features of the different classes.

### Model Evaluation

The primary metric used to evaluate the model's performance was the loss value, specifically the cross-entropy loss over validation data. While loss values offer direct insight into the model's learning process, future work may benefit from incorporating additional metrics such as F1-score for a more accurate assessment of segmentation accuracy.

### Training

The training process was optimised for computational efficiency by employing PyTorch's Automatic Mixed Precision (AMP) and gradient checkpointing. These techniques, alongside a carefully chosen batch size of 6, allowed the model to train effectively within the constraints of an 8GB GPU memory limit. The SGD optimiser, with a learning rate of approximately 0.04 and momentum of about 0.8 across 150 epochs, was calibrated to balance the model's learning speed and stability.

![Training Loss](https://raw.githubusercontent.com/JalalSayed1/Image-Damage-Classification/ff8499ff3aea505cd5880188e1392e3a19346f82/img/training_results/losses_lr%3D0.04491_momentum%3D0.83.png)

### Architecture Modifications

The original U-Net architecture was modified by simplifying double convolutional layers to single layers to reduce model complexity for more efficient training. Additionally, the implementation of "use_checkpointing" in the U-Net architecture, combined with PyTorch's AMP, facilitated a significant reduction in memory usage. This strategic adjustment ensured that the training process was memory-efficient and allowed for an increase in batch size from 4 to 6, improving the training efficiency.

## Results and Observations

Visual comparisons between the original images, ground truth annotations, and model predictions underscore the challenges the model faces in accurately segmenting all images and misclassifying non-background elements as background.

![Example final Predictions](https://raw.githubusercontent.com/JalalSayed1/Image-Damage-Classification/master/img/predictions/final_predictions.png)

### Discussion and Future Work

The challenges encountered during this project reveal significant insights into the limitations and potential areas for improvement in the semantic segmentation model. Key observations and strategic recommendations for future work are outlined below:

- Predominance of Background Classifications: The model frequently misclassified diverse image regions as background. This issue underscores the model's struggle with differentiating foreground and background classes, potentially due to several underlying factors discussed below.

- Class imbalance: The dataset is highly imbalanced, with the background class being the most common class. This causes the model to be biased towards the background class and struggle to learn the features of the other classes. Even by ignoring the background class in the loss function, the model still struggled to learn the features of the different classes.

- Small dataset: The dataset is relatively small. This can make it difficult for the model to learn the features of the different classes and can lead to overfitting. A pre-trained model may have been more effective for this small dataset.

- Simplified Model Complexity: Though beneficial for training efficiency, the decision to simplify the U-Net architecture may have compromised the model's capacity to capture complex features. Reintroducing complexity in a controlled manner with additional layers and deeper architecture could enhance feature extraction and segmentation accuracy.

- Data augmentation: The dataset was not augmented before training. Data augmentation can be an effective way to increase the diversity of the dataset and help the model learn the features of the different classes more effectively.

- Different Loss Function and Optimiser: Using a different loss function, such as Dice loss or focal loss, may have led to better results. I used cross-entropy loss as it is a popular choice for semantic segmentation tasks, but other loss functions may have been more effective for this task (e.g. focal loss for class imbalance or dice loss for small objects). Similarly, using a different optimiser like Adam may have led to better results. I used SGD as it is a popular choice for semantic segmentation tasks, but other optimisations may have been more effective for this task.

- Different batch size: Different sizes may have led to better results. I used a batch size of 6 as my GPU memory was limited.

- Different architecture: Using a different architecture, such as FCN or SegNet, may have led to better results. UNet is an excellent choice for semantic segmentation tasks, but other architectures may have been more effective for this task.

## Conclusion

In conclusion, the U-Net architecture was used to train a model to perform semantic segmentation on a dataset of damaged images. The model was trained using the SGD optimiser with a learning rate of ~0.04 and momentum of ~0.8 for 150 epochs. The model struggled to learn the features of the different classes and could not predict correct annotations for all the images. The reasons for this may include class imbalance, small dataset, model complexity, hyperparameters, data augmentation, loss function, optimiser, batch size, and architecture. Future work could address these issues and experiment with different approaches to improve the model's performance.

---
