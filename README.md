# DL_CW
COMPSCI5085 Deep Learning M Coursework


## Notes
1. Design, implement and test a deep learning architecture to detect and identify damage in images.
2. Each colour in those images corresponds to a different category of damage, including fold, writing or burn marks. 
3. can decide to use unsupervised pre-training of only supervised end-to-end training.
4. Dealing with a semantic segmentation problem, where the goal is to classify each pixel of an image into a predefined category.
5. Convolutional Neural Networks (CNNs) are typically the architecture of choice due to their effectiveness in image recognition and classification tasks.

#### Colour labels for each damage type

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

