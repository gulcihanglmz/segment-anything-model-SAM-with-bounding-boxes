# SAM Segmentation with Bounding Boxes

This script uses the **Segment Anything Model (SAM)** to perform semantic segmentation on images based on bounding box annotations provided in XML files. The results are saved as overlayed images showing the segmented regions.

## Features
- Loads images and their corresponding bounding box annotations.
- Uses SAM to generate segmentation masks for the specified bounding boxes.
- Saves the segmented images with visualized masks in the output folder.

## Requirements
- Python 3.8 or later
- Required libraries:
  - `torch`
  - `numpy`
  - `opencv-python`
  - `xml.etree.ElementTree`
  - `segment-anything`

## Installation
1. Clone or download this script.
2. Install the required Python libraries:
   ```bash
   pip install torch numpy opencv-python segment-anything
```bash

dataset/
├── JPEGImages/      # Contains the image files (.jpg, .png)
└── Annotations/     # Contains the corresponding XML files with bounding boxes
```


## Examples 

<img src="https://github.com/user-attachments/assets/17d3337d-b0df-46ac-b433-d7845a08b50b" width="300" height="200">
<img src="https://github.com/user-attachments/assets/94264519-f9cb-408a-b141-a0c9592a3a76" width="300" height="200">
<img src="https://github.com/user-attachments/assets/ed88df96-dd41-425c-9453-317d25bf9d31" width="300" height="200">


