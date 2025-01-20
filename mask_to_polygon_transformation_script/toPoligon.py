import os
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def process_all_masks(root_dir, output_root, class_mapping):
    """
    Process all masks from train, valid, and test subdirectories.
    Converts masks into YOLO-compatible polygon TXT format.

    Args:
    - root_dir (str): Root directory containing 'train', 'test', 'valid' folders.
    - output_root (str): Root directory to save output labels and images.
    - class_mapping (dict): Mapping from pixel values to class IDs.
    """
    for split in ['train', 'test', 'valid']:
        mask_dir = os.path.join(root_dir, split)
        output_dir = os.path.join(output_root, split, 'labels')
        os.makedirs(output_dir, exist_ok=True)

        image_dir = os.path.join(root_dir, split)
        output_image_dir = os.path.join(output_root, split, 'images')
        os.makedirs(output_image_dir, exist_ok=True)

        for file_name in os.listdir(mask_dir):
            if file_name.endswith('_mask.png'):  # Mask dosyalarını seç
                mask_path = os.path.join(mask_dir, file_name)
                label_file = file_name.replace('_mask.png', '.txt')  # TXT dosya ismi
                output_label_path = os.path.join(output_dir, label_file)

                # Maskeyi işle ve poligonlara çevir
                convert_mask_to_polygons(mask_path, output_label_path, class_mapping)

                # Orijinal görüntüyü da aynı isimle kaydet
                image_file = file_name.replace('_mask.png', '.jpg')
                original_image_path = os.path.join(image_dir, image_file)
                output_image_path = os.path.join(output_image_dir, image_file)

                # Orijinal resmi kopyala
                if os.path.exists(original_image_path):
                    Image.open(original_image_path).save(output_image_path)


def convert_mask_to_polygons(mask_path, output_label_path, class_mapping):
    """
    Convert a single mask to YOLO-compatible polygon TXT format.

    Args:
    - mask_path (str): Path to the mask file (.png).
    - output_label_path (str): Path to save the output TXT file.
    - class_mapping (dict): Mapping from pixel values to class IDs.
    """
    mask = np.array(Image.open(mask_path))
    with open(output_label_path, 'w') as f_out:
        for pixel_value, class_id in class_mapping.items():
            binary_mask = (mask == pixel_value).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue  # Eğer sınıfa ait piksel yoksa atla

            # Maskeyi RLE formatına çevir ve poligonları üret
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')  # Uyumluluk için
            polygons = mask_utils.toBbox(rle)  # Bounding box üret

            # YOLO formatında kaydet
            f_out.write(f"{class_id} " + " ".join(map(str, polygons)) + "\n")


# Kullanım
root_directory = "D:\segmentation\mask-semantic"  # 'segment' ana klasörü
output_directory = "D:\segmentation\polygon_images"  # Çıkış etiketlerinin ve resimlerin kaydedileceği ana klasör
class_id_map = {0: 0, 1: 1}  # Pixel değerlerini sınıf ID'lerine eşleştir
process_all_masks(root_directory, output_directory, class_id_map)
