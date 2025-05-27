import json
import os
import shutil
from pathlib import Path
import argparse
import random
from PIL import Image, ImageDraw, ImageFont

# Define a few colors for bounding boxes, cycling through them
BOX_COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "cyan"]

def draw_debug_visualizations_for_split(split_images_data, split_annotations_data, yolo_image_source_dir, coco_categories_list, num_samples_to_draw, debug_image_height=720):
    """
    Generates debug visualizations for a given split.
    Selects random images, draws their bounding boxes and category labels.
    """
    drawn_pil_images = []
    if not split_images_data:
        return drawn_pil_images

    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_categories_list}

    # Filter images that have annotations
    image_ids_with_annotations = set(ann['image_id'] for ann in split_annotations_data)
    eligible_images_info = [img for img in split_images_data if img['id'] in image_ids_with_annotations]

    if not eligible_images_info:
        print(f"No images with annotations found in this split for debugging.")
        return drawn_pil_images

    num_to_sample = min(num_samples_to_draw, len(eligible_images_info))
    sampled_images_info = random.sample(eligible_images_info, num_to_sample)

    default_font_size = 80
    try:
        font = ImageFont.truetype("Fonts/SimSun.ttf", default_font_size)
    except IOError:
        font = ImageFont.load_default()
        actual_font_size = 10 # load_default font is small
        print("Arial font not found, using default font. Text quality may vary.")


    for img_info in sampled_images_info:
        image_path = yolo_image_source_dir / img_info['file_name']
        if not image_path.exists():
            print(f"Debug: Image file {image_path} not found. Skipping this sample.")
            continue

        try:
            pil_image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(pil_image)

            img_annotations = [ann for ann in split_annotations_data if ann['image_id'] == img_info['id']]

            for i, ann in enumerate(img_annotations):
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                category_name = cat_id_to_name.get(category_id, f"ID:{category_id}")

                x, y, w, h = bbox
                shape = [(x, y), (x + w, y + h)]
                box_color = BOX_COLORS[i % len(BOX_COLORS)]
                draw.rectangle(shape, outline=box_color, width=3)
                
                text_position = (x + 5, y + 5)
                # Simple text background for better visibility
                text_bbox = draw.textbbox(text_position, category_name, font=font)
                draw.rectangle(text_bbox, fill=box_color)
                draw.text(text_position, category_name, fill="white", font=font)


            # Add filename as title
            title_text = img_info['file_name']
            title_font_size = 100
            try:
                title_font = ImageFont.truetype("Fonts/SimSun.ttf", title_font_size)
            except IOError:
                title_font = ImageFont.load_default() # Fallback for title
            
            title_position = (10,10)
            title_text_bbox = draw.textbbox(title_position, title_text, font=title_font)
            draw.rectangle(title_text_bbox, fill="black")
            draw.text(title_position, title_text, fill="white", font=title_font)


            # Resize for consistent height in composite image, maintaining aspect ratio
            original_width, original_height = pil_image.size
            aspect_ratio = original_width / original_height
            new_width = int(debug_image_height * aspect_ratio)
            pil_image = pil_image.resize((new_width, debug_image_height), Image.Resampling.LANCZOS)

            drawn_pil_images.append(pil_image)
            print(f"Debug: Drew {len(img_annotations)} annotations on {img_info['file_name']}")

        except Exception as e:
            print(f"Error processing image {image_path} for debug: {e}")
            
    return drawn_pil_images

def create_composite_debug_image(pil_images_list, output_path_str, canvas_width=3840, canvas_height=2160, images_per_row=3):
    """
    Creates a composite 4K image from a list of PIL images.
    """
    if not pil_images_list:
        print("Debug: No images to create a composite debug image.")
        return

    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    composite_image = Image.new('RGB', (canvas_width, canvas_height), (20, 20, 20)) # Dark gray background

    if not pil_images_list:
        print("No images provided for composite image.")
        composite_image.save(output_path, "JPEG")
        return

    # Calculate cell size (assuming all debug images were resized to a common height)
    # We'll primarily use the height of the first image as a guide for row height
    # and then fit `images_per_row` images.
    
    padding = 20 # Padding around images and canvas edge
    
    # Determine max width per image in a row to fit images_per_row
    available_width_for_images = canvas_width - (images_per_row + 1) * padding
    max_img_width_in_row = available_width_for_images // images_per_row
    
    # Use height of first (resized) image to determine row height.
    # All images should have been resized to same height by draw_debug_visualizations_for_split
    sample_img_height = pil_images_list[0].height if pil_images_list else 720 # default if list is empty, though checked before
    
    num_rows = (len(pil_images_list) + images_per_row - 1) // images_per_row
    available_height_for_images = canvas_height - (num_rows + 1) * padding
    max_img_height_in_col = available_height_for_images // num_rows

    # Final target height for each image in the composite, ensuring it fits
    target_img_height = min(sample_img_height, max_img_height_in_col)


    current_x = padding
    current_y = padding
    row_tallest_image = 0

    for i, img in enumerate(pil_images_list):
        # Resize again to fit the calculated cell, maintaining aspect ratio based on target_img_height
        aspect = img.width / img.height
        new_h = target_img_height
        new_w = int(new_h * aspect)

        if new_w > max_img_width_in_row: # If it's too wide, scale by width instead
            new_w = max_img_width_in_row
            new_h = int(new_w / aspect)
        
        img_to_paste = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if current_x + img_to_paste.width + padding > canvas_width: # Move to next row
            current_x = padding
            current_y += row_tallest_image + padding
            row_tallest_image = 0
        
        if current_y + img_to_paste.height + padding > canvas_height: # Out of canvas space
            print(f"Warning: Not enough space on composite image for all samples. Stopping at image {i+1}.")
            break

        composite_image.paste(img_to_paste, (current_x, current_y))
        current_x += img_to_paste.width + padding
        row_tallest_image = max(row_tallest_image, img_to_paste.height)
        
        if (i + 1) % images_per_row == 0: # End of a row
            current_x = padding
            current_y += row_tallest_image + padding
            row_tallest_image = 0


    composite_image.save(output_path, "JPEG", quality=90)
    print(f"Debug: Composite debug image saved to {output_path}")


def create_coco_splits(main_coco_path_str, yolo_dir_str, output_dir_str, debug_enabled=False, num_debug_samples_per_split=2):
    main_coco_path = Path(main_coco_path_str)
    yolo_dir = Path(yolo_dir_str)
    output_dir = Path(output_dir_str)

    print(f"Main COCO JSON: {main_coco_path}")
    print(f"YOLO data directory: {yolo_dir}")
    print(f"Output COCO directory: {output_dir}")
    if debug_enabled:
        print(f"Debug mode enabled. Will generate visualizations for {num_debug_samples_per_split} samples per split.")

    # 1. Create output directories
    output_annotations_dir = output_dir / "annotations"
    output_train_images_dir = output_dir / "train2017"
    output_val_images_dir = output_dir / "val2017"
    output_test_images_dir = output_dir / "test2017"
    
    debug_visualization_output_dir = None
    if debug_enabled:
        debug_visualization_output_dir = output_dir / "debug"
        os.makedirs(debug_visualization_output_dir, exist_ok=True)


    os.makedirs(output_annotations_dir, exist_ok=True)
    os.makedirs(output_train_images_dir, exist_ok=True)
    os.makedirs(output_val_images_dir, exist_ok=True)
    os.makedirs(output_test_images_dir, exist_ok=True)
    print("Created output directories.")

    # 2. Read category names
    category_names_path = yolo_dir / "category.names"
    coco_categories = []
    try:
        with open(category_names_path, 'r') as f:
            for i, line in enumerate(f):
                name = line.strip()
                if name:
                    coco_categories.append({"id": i + 1, "name": name, "supercategory": ""})
        print(f"Loaded {len(coco_categories)} categories: {coco_categories}")
    except FileNotFoundError:
        print(f"ERROR: Category names file not found at {category_names_path}")
        return

    # 3. Load the main COCO JSON
    try:
        with open(main_coco_path, 'r') as f:
            main_coco_data = json.load(f)
        print(f"Loaded main COCO JSON from {main_coco_path}")
    except FileNotFoundError:
        print(f"ERROR: Main COCO JSON not found at {main_coco_path}")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode main COCO JSON from {main_coco_path}")
        return

    main_images_by_filename = {img['file_name']: img for img in main_coco_data.get('images', [])}
    
    main_annotations_by_orig_image_id = {}
    # original_image_id_to_filename = {img['id']: img['file_name'] for img in main_coco_data.get('images', [])} # Not used

    for ann in main_coco_data.get('annotations', []):
        original_img_id = ann['image_id']
        if original_img_id not in main_annotations_by_orig_image_id:
            main_annotations_by_orig_image_id[original_img_id] = []
        main_annotations_by_orig_image_id[original_img_id].append(ann)

    # 4. Process each split
    splits_to_process = [
        ("train", yolo_dir / "train.txt", yolo_dir / "train", output_train_images_dir),
        ("val", yolo_dir / "val.txt", yolo_dir / "val", output_val_images_dir),
        ("test", yolo_dir / "test.txt", yolo_dir / "test", output_test_images_dir),
    ]

    base_info = main_coco_data.get("info", {"year": 2023, "version": "1.0", "description": "Split COCO Dataset", "contributor": "Script", "url": "", "date_created": ""})
    base_licenses = main_coco_data.get("licenses", [{"id": 1, "url": "", "name": "Unknown"}])

    all_pil_images_for_composite_debug = []

    for split_name, yolo_list_file, yolo_image_source_dir, split_output_images_dir in splits_to_process:
        print(f"\\nProcessing split: {split_name}")
        
        split_coco_images = []
        split_coco_annotations = []
        new_image_id_counter = 0
        new_annotation_id_counter = 0

        if not yolo_list_file.exists():
            print(f"Warning: YOLO list file for {split_name} not found at {yolo_list_file}. Skipping this split.")
            continue

        with open(yolo_list_file, 'r') as f_list:
            for line_number, line in enumerate(f_list):
                base_filename = line.strip()
                # If base_filename is an absolute path, extract just the filename
                base_filename = os.path.basename(base_filename.strip())
                if not base_filename:
                    continue

                if base_filename in main_images_by_filename:
                    original_image_info = main_images_by_filename[base_filename]
                    original_image_id = original_image_info['id'] # ID from the original coco_detection.json

                    # Create new image entry for the split
                    new_image_entry = original_image_info.copy()
                    new_image_entry['id'] = new_image_id_counter # This is the new ID for this split's JSON
                    split_coco_images.append(new_image_entry)

                    # Copy image file
                    source_image_path = yolo_image_source_dir / base_filename
                    dest_image_path = split_output_images_dir / base_filename
                    if source_image_path.exists():
                        shutil.copy(source_image_path, dest_image_path)
                    else:
                        print(f"Warning: Source image {source_image_path} not found for {base_filename} in {split_name} split.")

                    # Add annotations for this image
                    if original_image_id in main_annotations_by_orig_image_id:
                        for original_ann in main_annotations_by_orig_image_id[original_image_id]:
                            new_ann_entry = original_ann.copy()
                            new_ann_entry['image_id'] = new_image_id_counter # Link to new image ID in this split
                            new_ann_entry['id'] = new_annotation_id_counter
                            # Ensure category_id is valid (present in our new coco_categories)
                            cat_exists = any(cat['id'] == new_ann_entry['category_id'] for cat in coco_categories)
                            if not cat_exists:
                                # print(f"Warning: Annotation ID {original_ann['id']} for image {base_filename} has category_id {new_ann_entry['category_id']} which is not in the loaded categories. Skipping this annotation.")
                                pass # Potentially noisy, can be enabled if needed
                                continue
                            split_coco_annotations.append(new_ann_entry)
                            new_annotation_id_counter += 1
                    
                    new_image_id_counter += 1
                else:
                    print(f"Warning: Image filename '{base_filename}' from {yolo_list_file} (line {line_number+1}) not found in main COCO JSON images.")

        # Write split COCO JSON
        output_split_json_path = output_annotations_dir / f"instances_{split_name}2017.json"
        split_coco_data_final = {
            "info": base_info,
            "licenses": base_licenses,
            "categories": coco_categories,
            "images": split_coco_images,
            "annotations": split_coco_annotations
        }
        with open(output_split_json_path, 'w') as f_json_out:
            json.dump(split_coco_data_final, f_json_out, indent=4)
        print(f"Saved {split_name} COCO JSON to {output_split_json_path} with {len(split_coco_images)} images and {len(split_coco_annotations)} annotations.")

        if debug_enabled and num_debug_samples_per_split > 0:
            print(f"Generating debug visualizations for {split_name} split...")
            # Pass split_coco_images and split_coco_annotations which contain the re-indexed IDs
            drawn_images = draw_debug_visualizations_for_split(
                split_images_data=split_coco_images, 
                split_annotations_data=split_coco_annotations,
                yolo_image_source_dir=yolo_image_source_dir, # Source for original images
                coco_categories_list=coco_categories,
                num_samples_to_draw=num_debug_samples_per_split
            )
            all_pil_images_for_composite_debug.extend(drawn_images)

    if debug_enabled and all_pil_images_for_composite_debug:
        composite_output_path = debug_visualization_output_dir / "debug.jpg"
        create_composite_debug_image(all_pil_images_for_composite_debug, str(composite_output_path))
    elif debug_enabled:
        print("Debug: No images were generated for the composite debug image.")


    print("\\nScript finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert and split a COCO dataset based on YOLO file lists.")
    parser.add_argument('--main_coco_json', type=str, required=True,
                        help='Path to the main (unsplit) COCO JSON annotation file.')
    parser.add_argument('--yolo_dir', type=str, required=True, 
                        help='Directory containing YOLO formatted data: category.names, train.txt, val.txt, test.txt, and train/val/test image subdirectories.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the output COCO 2017 formatted dataset. Defaults to a new directory named <yolo_dir_name>_coco next to yolo_dir.')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode. This will generate a composite image (debug.jpg) with 2 random annotated samples per split.')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir_param = Path(args.output_dir)
    else:
        yolo_path = Path(args.yolo_dir)
        output_dir_param = yolo_path.parent / (yolo_path.name + "_coco")
    
    Path(output_dir_param).mkdir(parents=True, exist_ok=True)

    num_debug_samples = 2 # As per user request "2 as function parameter" for debug mode

    create_coco_splits(
        main_coco_path_str=args.main_coco_json, 
        yolo_dir_str=args.yolo_dir, 
        output_dir_str=str(output_dir_param.resolve()),
        debug_enabled=args.debug,
        num_debug_samples_per_split=num_debug_samples if args.debug else 0
    ) 