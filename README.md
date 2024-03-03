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
7. To monitor GPU usage: `nvidia-smi -l 5 --query-gpu=utilization.gpu,memory.used --format=csv,nohea
der,nounits`

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

## Ideas

1. Segmentation?
2. How to save and load the model: [https://chat.openai.com/share/147c1c0d-e8a6-4ec1-8afe-54eaad04de40]
3. Predict the annotation

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
5. 
