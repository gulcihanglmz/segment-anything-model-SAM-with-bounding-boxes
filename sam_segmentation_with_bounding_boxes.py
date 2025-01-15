import os
import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from segment_anything import SamPredictor, sam_model_registry

# 1. Load the SAM Model
MODEL_PATH = "D:\\segmentation\\sam_vit_b_01ec64.pth"  # Full path to the model file
device = "cpu"
sam = sam_model_registry["vit_b"](MODEL_PATH)
sam.to(device)
sam_predictor = SamPredictor(sam)

# 2. Load Image
def load_image(image_path):
    """Loads an image and converts it to RGB format."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"File not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# 3. Read XML Label File
def read_labels_from_xml(xml_path):
    """Reads bounding boxes from an XML file."""
    bounding_boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                bounding_boxes.append([xmin, ymin, xmax, ymax])
    except Exception as e:
        print(f"Error occurred while processing XML file ({xml_path}): {e}")
    return bounding_boxes

# 4. Segment Specific Regions with SAM
def segment_with_boxes(image_path, bounding_boxes, output_path):
    """Performs segmentation using specific bounding boxes and saves the output file."""
    image = load_image(image_path)
    sam_predictor.set_image(image)
    overlay = image.copy()

    for box in bounding_boxes:
        try:
            # Convert bounding box to a numpy array and pass it to the predictor
            box = np.array(box)

            # Check bounding box validity
            if box[0] < 0 or box[1] < 0 or box[2] > image.shape[1] or box[3] > image.shape[0]:
                print(f"Invalid bounding box skipped: {box}")
                continue

            masks, _, _ = sam_predictor.predict(
                box=box,
                point_coords=None,
                point_labels=None,
                multimask_output=False
            )
            mask = masks[0]

            # Check mask validity
            if mask.sum() == 0:
                print(f"No mask found for this bounding box: {box}")
                continue

            # Apply the segmentation mask to the overlay
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5
        except Exception as e:
            print(f"Error occurred during segmentation ({box}): {e}")
            continue

    # Save the image
    result_image = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)
    print(f"Segmentation result saved: {output_path}")

# 5. Main Function
def process_folder(dataset_folder, output_folder):
    """Processes all image and label files in the dataset folder."""
    image_folder = os.path.join(dataset_folder, "JPEGImages")
    annotation_folder = os.path.join(dataset_folder, "Annotations")

    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(image_folder):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(image_folder, file_name)
            label_path = os.path.join(annotation_folder, os.path.splitext(file_name)[0] + ".xml")
            output_path = os.path.join(output_folder, file_name)

            if not os.path.exists(label_path):
                print(f"Label file not found: {label_path}, skipping.")
                continue

            try:
                bounding_boxes = read_labels_from_xml(label_path)
                segment_with_boxes(image_path, bounding_boxes, output_path)
            except Exception as e:
                print(f"Error occurred ({file_name}): {e}")

if __name__ == "__main__":
    dataset_folder = "D:\\segmentation\\dataset"  # Dataset folder
    output_folder = "D:\\segmentation\\output"  # Output folder

    try:
        process_folder(dataset_folder, output_folder)
        print("All files processed successfully!")
    except Exception as e:
        print(f"General error: {e}")
