import json
import os
from pathlib import Path
import argparse
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict
import copy # Added for deepcopy
from loguru import logger # Added for logging
from prettytable import PrettyTable # Added for table logging

# Define distinct colors for different categories
CATEGORY_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 128, 255),  # Light Blue
    (255, 0, 128),  # Pink
]

# English to Chinese category mapping
CATEGORY_MAPPING = {
    'FLIGHT': '打架',
    'TURNLEFT': '左转',
    'PARKING': '停车',
    'TURNROUND': '转弯'
}

def load_coco_dataset(coco_json_path):
    """Load COCO dataset from JSON file."""
    try:
        with open(coco_json_path, 'r') as f:
            data = json.load(f)
        if not data.get('categories'):
            print(f"Warning: No categories found in {coco_json_path}")
            return None
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {coco_json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {coco_json_path}")
        return None
    except Exception as e:
        print(f"Error loading {coco_json_path}: {str(e)}")
        return None

def get_category_colors(categories):
    """Assign colors to categories."""
    return {cat['id']: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] 
            for i, cat in enumerate(categories)}

def get_chinese_name(english_name):
    """Convert English category name to Chinese."""
    return CATEGORY_MAPPING.get(english_name.upper(), english_name)

def draw_annotations(image, annotations, category_colors, category_names, font=None):
    """Draw bounding boxes and labels on the image."""
    draw = ImageDraw.Draw(image)
    
    # Use provided font or try to load default/SimSun
    if font is None:
        try:
            font = ImageFont.truetype("Fonts/SimSun.ttf", 20) # Default size if no font passed
        except IOError:
            font = ImageFont.load_default()
    
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']
        color = category_colors[category_id]
        category_name = get_chinese_name(category_names[category_id])
        
        x, y, w, h = bbox
        shape = [(x, y), (x + w, y + h)]
        draw.rectangle(shape, outline=color, width=2)
        
        # Draw label background
        text = f"{category_name}"
        text_bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, y), text, fill="white", font=font)
    
    return image

def _log_placed_images_table(
    placed_image_details_for_current_canvas, 
    global_all_category_names_map, 
    global_sorted_category_ids, 
    splits_rendered_on_this_canvas_list
):
    if not placed_image_details_for_current_canvas:
        logger.info("No images were placed on the current canvas for detailed logging.")
        return

    # Group images by category and split
    category_split_images = defaultdict(lambda: defaultdict(list))
    for img_info in placed_image_details_for_current_canvas:
        primary_cat_name = get_chinese_name(global_all_category_names_map[img_info['primary_cat_id']])
        category_split_images[primary_cat_name][img_info['split_name']].append(img_info)

    # Get sorted list of Chinese category names as they appear on the Y-axis
    sorted_cat_names_chinese = [get_chinese_name(global_all_category_names_map[cat_id]) for cat_id in global_sorted_category_ids]

    # Create header
    logger.info("\n=== Placed Images on Current Canvas ===")
    header = "Category".ljust(10) + " | "
    for split in splits_rendered_on_this_canvas_list:
        header += split.upper().ljust(15) + " | "
    logger.info(header)
    logger.info("-" * len(header))

    # Print each category's images
    for cat_name in sorted_cat_names_chinese:
        row = cat_name.ljust(10) + " | "
        for split in splits_rendered_on_this_canvas_list:
            images = category_split_images[cat_name][split]
            if images:
                # Sort images by filename
                images.sort(key=lambda x: x['file_name'])
                # Create image list string
                img_list = ", ".join([f"{img['file_name']} ({img['w']}x{img['h']})" for img in images])
                row += img_list.ljust(15) + " | "
            else:
                row += "-".ljust(15) + " | "
        logger.info(row)

    logger.info("=== End of Placed Images ===\n")

def calculate_canvas_dimensions(
    num_categories,
    num_splits,
    layout_margin,
    layout_category_width,
    layout_cell_height,
    max_images_per_category,
    sub_image_short_length,
    layout_image_spacing,
    current_split_images_count,  # For individual split visualization
    category_images_count=None   # For category-wise image count
):
    """Calculate canvas dimensions based on content."""
    # Calculate height based on number of categories and cell height
    canvas_height = layout_margin * 2 + num_categories * layout_cell_height
    
    # Calculate width based on content
    split_width = layout_category_width + layout_margin
    
    if category_images_count is not None:
        # For combined visualization, calculate width based on total images across all splits
        total_images = current_split_images_count
        content_width = total_images * (sub_image_short_length + layout_image_spacing) + sub_image_short_length * 2
    else:
        # For individual split visualization, use the current split's image count
        content_width = current_split_images_count * (sub_image_short_length + layout_image_spacing) + sub_image_short_length * 2
    
    canvas_width = layout_margin * 2 + num_splits * split_width + content_width
    
    return canvas_width, canvas_height

def _draw_splits_on_canvas(
    draw_context, # PIL.ImageDraw.Draw instance (with .canvas attribute referring to the Image)
    splits_to_render_on_this_canvas, 
    all_prepared_images_master_list, 
    all_category_names_map, # global map of ID to English name
    category_id_to_row_index_map, # global map of ID to Y-axis row index
    sorted_global_cat_ids, # global list of category IDs, sorted for Y-axis display
    layout_margin, layout_category_width, layout_cell_padding, layout_image_spacing,
    effective_split_col_width_on_this_canvas, 
    sub_image_short_length, # Parameter for image scaling
    layout_cell_height,
    font_title,
    category_colors_map  # Add category colors map parameter
):
    
    placed_images_for_current_canvas = []
    images_placed_count_on_this_canvas = 0
    content_area_end_x = 0

    # Calculate total width needed for all splits
    total_splits_width = 0
    for split_idx, split_name in enumerate(splits_to_render_on_this_canvas):
        images_in_split = [img for img in all_prepared_images_master_list if img['split_name'] == split_name]
        split_width = len(images_in_split) * (sub_image_short_length + layout_image_spacing)
        total_splits_width += split_width

    # Calculate base position for the first split
    current_split_base_x_on_canvas = layout_margin + layout_category_width

    for split_idx, split_name in enumerate(splits_to_render_on_this_canvas):
        # Get and scale images for this split
        images_to_process_for_this_split = copy.deepcopy([
            img for img in all_prepared_images_master_list if img['split_name'] == split_name and img['annotated_full_image'] is not None
        ])

        # Calculate content area boundaries for this split
        content_area_start_x = current_split_base_x_on_canvas + layout_cell_padding
        
        # Calculate width needed for this split based on its images
        split_width = len(images_to_process_for_this_split) * (sub_image_short_length + layout_image_spacing)
        content_area_end_x = content_area_start_x + split_width

        # Draw split header
        header_text_bbox = draw_context.textbbox((0,0), split_name.upper(), font=font_title)
        header_text_width = header_text_bbox[2] - header_text_bbox[0]
        header_x = current_split_base_x_on_canvas + (split_width - header_text_width) / 2
        draw_context.text((header_x, layout_margin // 2), 
                          split_name.upper(), fill=(0, 0, 0), font=font_title)

        # If no images to process, skip to next split
        if not images_to_process_for_this_split:
            logger.info(f"No images to process for split '{split_name}', skipping.")
            continue

        # Scale images based on short_length parameter
        for img_data in images_to_process_for_this_split:
            w_ann, h_ann = img_data['annotated_w'], img_data['annotated_h']
            if w_ann == 0 or h_ann == 0:
                img_data['current_scaled_w'] = 0; img_data['current_scaled_h'] = 0; continue
            
            # Scale based on the shorter side to maintain aspect ratio
            if w_ann < h_ann: # Portrait or square, short side is width
                scale = sub_image_short_length / w_ann
                img_data['current_scaled_w'] = sub_image_short_length
                img_data['current_scaled_h'] = int(h_ann * scale)
            else: # Landscape or square, short side is height
                scale = sub_image_short_length / h_ann
                img_data['current_scaled_h'] = sub_image_short_length
                img_data['current_scaled_w'] = int(w_ann * scale)
        
        images_to_process_for_this_split = [img for img in images_to_process_for_this_split if img.get('current_scaled_w', 0) > 0 and img.get('current_scaled_h', 0) > 0]

        # If no valid images after scaling, skip to next split
        if not images_to_process_for_this_split:
            logger.info(f"No valid images after scaling for split '{split_name}', skipping.")
            continue

        # Group images by their primary category
        images_by_category = defaultdict(list)
        for img_data in images_to_process_for_this_split:
            primary_cat_id = None
            min_cat_idx = float('inf')
            # Sort categories of the image by their global display order
            img_categories_sorted_by_global_order = sorted(
                list(img_data['categories']), 
                key=lambda cid: sorted_global_cat_ids.index(cid) if cid in sorted_global_cat_ids else float('inf')
            )
            if img_categories_sorted_by_global_order:
                _primary_cat_id_candidate = img_categories_sorted_by_global_order[0]
                if _primary_cat_id_candidate in category_id_to_row_index_map:
                    primary_cat_id = _primary_cat_id_candidate
                    min_cat_idx = category_id_to_row_index_map[primary_cat_id]
            
            if primary_cat_id is not None:
                images_by_category[primary_cat_id].append({**img_data, 'primary_cat_id': primary_cat_id, 'primary_cat_order_idx': min_cat_idx})
        
        # Sort images within each category by total annotations
        for cat_id in images_by_category:
            images_by_category[cat_id].sort(key=lambda x: x['total_annotations'])

        # Place images category by category
        current_x_offset = content_area_start_x
        
        # Process each category in order
        for cat_id in sorted_global_cat_ids:
            if cat_id not in images_by_category:
                continue
                
            images_in_category = images_by_category[cat_id]
            cat_row_idx = category_id_to_row_index_map[cat_id]
            
            for image_to_place in images_in_category:
                final_w = image_to_place['current_scaled_w']
                final_h = image_to_place['current_scaled_h']

                if final_w <= 0 or final_h <= 0:
                    logger.warning(f"Image {image_to_place['file_name']} resulted in zero/negative dimension ({final_w}x{final_h}) after scaling. Skipping.")
                    continue

                try:
                    img_obj_to_paste = image_to_place['annotated_full_image'].resize((final_w, final_h), Image.Resampling.LANCZOS)
                except Exception as e: 
                    logger.error(f"Error resizing image {image_to_place['file_name']} to final dimensions ({final_w}x{final_h}): {str(e)}", exc_info=True); continue

                # Calculate image position
                actual_x_on_canvas = current_x_offset
                actual_y_on_canvas = layout_margin + cat_row_idx * layout_cell_height + (layout_cell_height - final_h) / 2 # Center vertically in cell
                
                # Check if the image fits in the content area
                if actual_x_on_canvas + final_w <= content_area_end_x:
                    draw_context.canvas.paste(img_obj_to_paste, (int(actual_x_on_canvas), int(actual_y_on_canvas)))
                    images_placed_count_on_this_canvas += 1
                    
                    # Draw colored squares for secondary categories
                    square_size = sub_image_short_length  # Use short_length for square size
                    square_spacing = 5  # Small spacing between squares
                    current_square_x = actual_x_on_canvas  # Start after the image
                    
                    # Sort secondary categories by their global display order
                    secondary_cats = sorted(
                        [cat_id for cat_id in image_to_place['categories'] if cat_id != image_to_place['primary_cat_id']],
                        key=lambda cid: sorted_global_cat_ids.index(cid) if cid in sorted_global_cat_ids else float('inf')
                    )
                    
                    for sec_cat_id in secondary_cats:
                        if sec_cat_id in category_id_to_row_index_map:
                            sec_cat_row_idx = category_id_to_row_index_map[sec_cat_id]
                            sec_cat_y = layout_margin + sec_cat_row_idx * layout_cell_height + (layout_cell_height - square_size) / 2
                            
                            # Draw the colored square
                            color = category_colors_map.get(sec_cat_id, (0, 0, 0))
                            draw_context.rectangle(
                                [current_square_x, sec_cat_y, 
                                 current_square_x + square_size, sec_cat_y + square_size],
                                fill=color
                            )
                    
                    # Store details of the placed image for multi-category highlighting and logging
                    placed_images_for_current_canvas.append({
                        'id': image_to_place['id'], 'file_name': image_to_place['file_name'],
                        'split_name': split_name, 'x': actual_x_on_canvas, 'y': actual_y_on_canvas, 
                        'w': final_w, 'h': final_h, 'active_cats': image_to_place['categories'], 
                        'primary_cat_id': image_to_place['primary_cat_id'] # Store primary_cat_id for logging
                    })
                    current_x_offset += final_w + layout_image_spacing
                else:
                    logger.warning(f"Image {image_to_place['file_name']} (final_w={final_w}) would overflow split '{split_name}' content area (max_x={content_area_end_x}, current_x={actual_x_on_canvas}). Stopping further placement in this split.")
                    break # No more images will fit horizontally in this split column's content area

        # Update base position for next split
        current_split_base_x_on_canvas = content_area_end_x + layout_margin
    
    logger.info(f"Placed {images_placed_count_on_this_canvas} images in total on the current canvas for splits: {', '.join(splits_to_render_on_this_canvas)}.")
    return placed_images_for_current_canvas

def create_visualization(coco_dir, output_path, specified_splits_to_draw=None, short_length=70):
    # Configure Loguru logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green>|L:<level>{level:<3}</level>| <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
    logger.add("logs/visualization_log.log", rotation="10 MB")
    
    logger.info(f"Starting visualization with parameters:")
    logger.info(f"- COCO directory: {coco_dir}")
    logger.info(f"- Output path: {output_path}")
    logger.info(f"- Specified splits: {specified_splits_to_draw if specified_splits_to_draw else 'All splits'}")
    logger.info(f"- Short length: {short_length}")

    # --- DATA LOADING AND CATEGORY PREPARATION ---
    default_split_names = ['train', 'val', 'test'] 
    datasets = {}
    all_categories_set = set()
    if not os.path.exists(coco_dir):
        logger.error(f"Coco dir not found: {coco_dir}")
        return
    annotations_dir = os.path.join(coco_dir, 'annotations')
    if not os.path.exists(annotations_dir):
        logger.error(f"Annotations dir not found: {annotations_dir}")
        return
    
    for s_name in default_split_names: 
        dataset = load_coco_dataset(os.path.join(annotations_dir, f'instances_{s_name}2017.json'))
        if dataset:
            datasets[s_name] = dataset
            all_categories_set.update(cat['id'] for cat in dataset.get('categories',[]))
    
    if not all_categories_set: 
        logger.error("No categories found in any dataset splits.")
        return
    
    temp_cat_names = {}
    for ds_name, ds_content in datasets.items():
        if ds_content and ds_content.get('categories'):
            for cat in ds_content['categories']:
                if cat['id'] not in temp_cat_names:
                    temp_cat_names[cat['id']] = cat['name']
    
    category_names_map = temp_cat_names
    valid_cat_ids = {cid for cid in all_categories_set if cid in category_names_map}
    if not valid_cat_ids: 
        logger.error("No valid categories with names found after filtering.")
        return
    
    sorted_category_ids = sorted(list(valid_cat_ids), key=lambda cid: category_names_map.get(cid, ''))
    category_id_to_row_index = {cat_id: i for i, cat_id in enumerate(sorted_category_ids)}
    category_colors_map = get_category_colors([{'id': cat_id} for cat_id in sorted_category_ids])

    # --- LAYOUT PARAMETERS ---
    margin, cat_width, img_spacing, cell_pad = 50, 250, 10, 5
    num_cat_rows = len(sorted_category_ids)
    if num_cat_rows == 0: 
        logger.error("No categories to draw.")
        return

    # Calculate cell height based on a reasonable default height
    default_canvas_height = 2160  # Use a default height for initial cell calculation
    cell_h = (default_canvas_height - margin * 2) // num_cat_rows
    if cell_h <= 0:
        logger.error(f"Calculated cell_height ({cell_h}) is too small or zero.")
        return

    # --- FONT LOADING ---
    try:
        font_p = "Fonts/SimSun.ttf"
        if not os.path.exists(font_p) and os.path.exists("NotoSansCJK-Regular.otf"): 
            font_p = "NotoSansCJK-Regular.otf"
            logger.info("Using NotoSansCJK-Regular.otf")
        elif not os.path.exists(font_p): 
            font_p = None
            logger.warning("SimSun.ttf and NotoSansCJK-Regular.otf not found. Using PIL default font.")
        
        title_f = ImageFont.truetype(font_p, 40) if font_p else ImageFont.load_default()
        cat_label_f = ImageFont.truetype(font_p, 30) if font_p else ImageFont.load_default()
        legend_f = ImageFont.truetype(font_p, 20) if font_p else ImageFont.load_default()
        ann_f = ImageFont.truetype(font_p, 15) if font_p else ImageFont.load_default()
    except Exception as e:
        logger.error("Font loading error", exc_info=True)
        title_f = cat_label_f = legend_f = ann_f = ImageFont.load_default()

    # --- PREPARE ALL IMAGE DATA (MASTER LIST) ---
    logger.info("Preparing all image data (load, annotate full size)...")
    all_prepared_images_master = []
    skipped_log = []
    
    for s_name in default_split_names:
        if s_name not in datasets: 
            logger.warning(f"Dataset for split '{s_name}' not loaded, skipping its images for master list.")
            continue
        dataset = datasets[s_name]
        for img_info in dataset.get('images', []):
            img_id = img_info['id']
            cats_for_img = {ann['category_id'] for ann in dataset.get('annotations',[]) 
                          if ann['image_id'] == img_id and ann['category_id'] in valid_cat_ids}
            if not cats_for_img: 
                skipped_log.append({'File':img_info['file_name'],'Split':s_name,'Reason':'No valid/selected cats for this image'})
                continue
            img_path = os.path.join(coco_dir, f'{s_name}2017', img_info['file_name'])
            if not os.path.exists(img_path):
                skipped_log.append({'File':img_info['file_name'],'Split':s_name,'Reason':'Image file not found'})
                continue
            try:
                orig_img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(orig_img)
                try:
                    # Create different font sizes for filename and category labels
                    filename_font = ImageFont.truetype("Fonts/SimSun.ttf", 40)  # Larger font for filename
                    category_font = ImageFont.truetype("Fonts/SimSun.ttf", 70)  # Even larger font for category labels
                except IOError:
                    filename_font = ImageFont.load_default()
                    category_font = ImageFont.load_default()
                
                # Draw filename at the top of the image with background
                filename = img_info['file_name']
                text_bbox = draw.textbbox((0, 0), filename, font=filename_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([(0, 0), (text_width + 10, text_height + 10)], fill=(0, 0, 0))
                draw.text((5, 5), filename, fill=(255, 255, 255), font=filename_font)
                
                # Draw annotations
                current_img_annotations = [ann for ann in dataset.get('annotations', []) 
                                        if ann['image_id'] == img_info['id']]
                if current_img_annotations:
                    # Get unique colors for categories
                    category_colors = {}
                    for ann in current_img_annotations:
                        if ann['category_id'] not in category_colors:
                            category_colors[ann['category_id']] = CATEGORY_COLORS[len(category_colors) % len(CATEGORY_COLORS)]
                    
                    # Draw annotations
                    for ann in current_img_annotations:
                        bbox = ann['bbox']
                        x, y, w, h = bbox
                        color = category_colors[ann['category_id']]
                        
                        # Draw bounding box
                        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=10)
                        
                        # Draw category name
                        cat_name = get_chinese_name(category_names_map.get(ann['category_id'], 'Unknown'))
                        label = f"{cat_name}"
                        text_bbox = draw.textbbox((0, 0), label, font=category_font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # Calculate label position to be above the bounding box
                        padding = 5  # Padding between label and box
                        label_y = max(0, y - text_height - padding)  # Ensure label doesn't go above image
                        text_y = label_y + padding  # Add small padding for text
                        
                        # Ensure label doesn't go beyond right edge of image
                        label_width = min(text_width + 10, orig_img.width - x)
                        
                        # Draw label background above the bounding box
                        draw.rectangle([(x, label_y), (x + label_width, y)], fill=color)
                        draw.text((x + padding, text_y), label, fill=(255, 255, 255), font=category_font)
                
                ann_img = draw_annotations(orig_img.copy(), current_img_annotations, 
                                        category_colors_map, category_names_map, font=ann_f)
                if ann_img.width == 0 or ann_img.height == 0: 
                    skipped_log.append({'File':img_info['file_name'],'Split':s_name,'Reason':'Annotated image has zero dimension'})
                    continue
                all_prepared_images_master.append({
                    'id': img_id, 'file_name': img_info['file_name'], 'split_name': s_name,
                    'categories': cats_for_img, 
                    'total_annotations': len(current_img_annotations),
                    'annotated_full_image': ann_img,
                    'annotated_w': ann_img.width, 
                    'annotated_h': ann_img.height
                })
            except Exception as e:
                logger.warning(f"Error processing image {img_info['file_name']} from split {s_name}",exc_info=True)
                skipped_log.append({'File':img_info['file_name'],'Split':s_name,'Reason':f'Exception: {str(e)}'})
    
    logger.info(f"Master image list preparation done. Loaded: {len(all_prepared_images_master)} images. Skipped during prep: {len(skipped_log)}")
    if skipped_log:
        sk_tbl=PrettyTable(); sk_tbl.field_names=["File","Split","Reason"]; [sk_tbl.add_row(r.values()) for r in skipped_log]
        logger.info(f"Skipped Images During Master List Preparation:\n{sk_tbl.get_string()}")

    # Calculate maximum images per category after preparing all images
    max_images_per_category = defaultdict(int)
    for img_data in all_prepared_images_master:
        for cat_id in img_data['categories']:
            max_images_per_category[cat_id] += 1
    max_images_per_cat = max(max_images_per_category.values()) if max_images_per_category else 0

    # --- DETERMINE DRAWING MODE AND LOOP TARGETS ---
    single_image_per_split_mode = False
    if specified_splits_to_draw:
        splits_for_drawing_loop = [s for s in specified_splits_to_draw if s in datasets]
        if not splits_for_drawing_loop:
            logger.error(f"None of the specified splits for drawing ({specified_splits_to_draw}) are valid or have loaded data. Exiting.")
            return
        single_image_per_split_mode = True
        logger.info(f"Mode: Individual image per specified split for {splits_for_drawing_loop}")
    else:
        splits_for_drawing_loop = default_split_names
        logger.info(f"Mode: Combined image for default splits {splits_for_drawing_loop}")

    # --- MAIN DRAWING LOOP ---
    # First, generate individual split visualizations if in single_image_per_split_mode
    if single_image_per_split_mode:
        loop_targets_for_canvas_creation = splits_for_drawing_loop
        for current_run_item in loop_targets_for_canvas_creation:
            # Determine which splits to render on THIS specific canvas iteration
            splits_to_render_on_this_canvas_now = [current_run_item]
            
            # Count images for current split
            current_split_images = [img for img in all_prepared_images_master if img['split_name'] == current_run_item]
            current_split_images_count = len(current_split_images)
            
            # Calculate dynamic canvas dimensions for this iteration
            num_splits = len(splits_to_render_on_this_canvas_now)
            canvas_width, canvas_height = calculate_canvas_dimensions(
                num_cat_rows,
                num_splits,
                margin,
                cat_width,
                cell_h,
                max_images_per_cat,
                short_length,
                img_spacing,
                current_split_images_count  # For individual split
            )

            logger.info(f"Calculated canvas dimensions: {canvas_width}x{canvas_height}")
            logger.info(f"Current split '{current_run_item}' has {current_split_images_count} images")
            logger.info(f"Rendering splits: {splits_to_render_on_this_canvas_now}")

            canvas = Image.new('RGB', (canvas_width, canvas_height), (0,0,0))
            draw = ImageDraw.Draw(canvas)
            draw.canvas = canvas

            # Draw static category labels on the left for the current canvas
            for i, cat_id in enumerate(sorted_category_ids):
                y_pos = margin + i * cell_h
                cat_name_chinese = get_chinese_name(category_names_map[cat_id])
                text_bbox = draw.textbbox((0,0), cat_name_chinese, font=cat_label_f)
                text_height = text_bbox[3] - text_bbox[1]
                text_y_centered = y_pos + (cell_h - text_height) / 2
                draw.text((margin // 2, text_y_centered), cat_name_chinese, fill=(0,0,0), font=cat_label_f)

            # Calculate column width for splits on THIS canvas
            effective_split_col_width = (canvas_width - cat_width - margin*2 - (num_splits-1)*margin) // num_splits if num_splits > 0 else 0
            
            if effective_split_col_width <= 0: 
                logger.error(f"Effective split column width ({effective_split_col_width}) is too small for current canvas with splits {splits_to_render_on_this_canvas_now}. Skipping this canvas.")
                continue

            placed_on_this_canvas_details = _draw_splits_on_canvas(
                draw, splits_to_render_on_this_canvas_now, 
                all_prepared_images_master,
                category_names_map, category_id_to_row_index, sorted_category_ids,
                margin, cat_width, cell_pad, img_spacing, 
                effective_split_col_width, 
                short_length,
                cell_h, title_f,
                category_colors_map  # Pass category colors map
            )

            # Multi-category highlighting for images placed on the current canvas
            # HACK: 多类别标记 (细实线)
            # MULTI_CAT_BOX_COLOR = "purple"
            # MULTI_CAT_BOX_WIDTH = 1
            # if placed_on_this_canvas_details:
            #     logger.info(f"Drawing {len(placed_on_this_canvas_details)} multi-cat highlights for current canvas.")
            #     for p_info in placed_on_this_canvas_details:
            #         if len(p_info['active_cats']) > 1:
            #             im_x,im_w = p_info['x'], p_info['w']
            #             for ac_id in p_info['active_cats']:
            #                 if ac_id not in category_id_to_row_index: continue
            #                 r_idx = category_id_to_row_index[ac_id]
            #                 hy0, hy1 = margin + r_idx*cell_h, margin+(r_idx+1)*cell_h
            #                 draw.rectangle((im_x,hy0,im_x+im_w,hy1),outline=MULTI_CAT_BOX_COLOR,width=MULTI_CAT_BOX_WIDTH)

            # Legend drawing for the current canvas
            lx,ly = canvas_width-cat_width-margin+20, margin
            for cid_legend in sorted_category_ids:
                draw.rectangle([lx,ly,lx+20,ly+20],fill=category_colors_map.get(cid_legend, (0,0,0)))
                draw.text((lx+30,ly),get_chinese_name(category_names_map.get(cid_legend, "N/A")),fill=(0,0,0),font=legend_f)
                ly+=30
            ly+=10
            # HACK: 多类别标记 (细实线)
            # draw.rectangle([lx,ly,lx+20,ly+20],outline=MULTI_CAT_BOX_COLOR,width=MULTI_CAT_BOX_WIDTH)
            # draw.text((lx+30,ly),"多类别标记 (细实线)",fill=(0,0,0),font=legend_f)

            # Log table for the current canvas
            _log_placed_images_table(placed_on_this_canvas_details, category_names_map, sorted_category_ids, splits_to_render_on_this_canvas_now)
            
            # Determine output path for the current canvas and save it
            current_canvas_output_path = output_path
            if single_image_per_split_mode:
                base_output_path = Path(output_path)
                current_canvas_output_path = base_output_path.parent / f"{base_output_path.stem}_{current_run_item}{base_output_path.suffix}"
            
            try:
                Path(current_canvas_output_path).parent.mkdir(parents=True, exist_ok=True)
                canvas.save(current_canvas_output_path, quality=95)
                logger.info(f"Successfully saved visualization:")
                logger.info(f"- Output path: {current_canvas_output_path}")
                logger.info(f"- Image dimensions: {canvas_width}x{canvas_height}")
                logger.info(f"- Number of categories: {num_cat_rows}")
                logger.info(f"- Number of splits: {num_splits}")
                logger.info(f"- Images placed: {len(placed_on_this_canvas_details)}")
            except Exception as e:
                logger.error(f"Error saving canvas to {current_canvas_output_path}", exc_info=True)

    # Then, always generate a combined visualization
    logger.info("\n=== Generating Combined Visualization ===")
    splits_to_render_on_this_canvas_now = splits_for_drawing_loop
    
    # Count images per category for all splits
    category_images_count = defaultdict(int)
    for img_data in all_prepared_images_master:
        for cat_id in img_data['categories']:
            if cat_id in category_id_to_row_index:
                category_images_count[cat_id] += 1
    
    logger.info("Images per category in combined visualization:")
    for cat_id, count in category_images_count.items():
        cat_name = get_chinese_name(category_names_map[cat_id])
        logger.info(f"- {cat_name}: {count} images")
    
    # Calculate dynamic canvas dimensions for combined visualization
    num_splits = len(splits_to_render_on_this_canvas_now)
    canvas_width, canvas_height = calculate_canvas_dimensions(
        num_cat_rows,
        num_splits,
        margin,
        cat_width,
        cell_h,
        max_images_per_cat,
        short_length,
        img_spacing,
        len(all_prepared_images_master),  # Total image count
        category_images_count  # Pass category-wise image count
    )

    logger.info(f"Calculated canvas dimensions for combined visualization: {canvas_width}x{canvas_height}")
    logger.info(f"Maximum images in any category row: {max(category_images_count.values()) if category_images_count else 0}")
    logger.info(f"Rendering all splits: {splits_to_render_on_this_canvas_now}")

    canvas = Image.new('RGB', (canvas_width, canvas_height), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    draw.canvas = canvas

    # Draw static category labels on the left
    for i, cat_id in enumerate(sorted_category_ids):
        y_pos = margin + i * cell_h
        cat_name_chinese = get_chinese_name(category_names_map[cat_id])
        text_bbox = draw.textbbox((0,0), cat_name_chinese, font=cat_label_f)
        text_height = text_bbox[3] - text_bbox[1]
        text_y_centered = y_pos + (cell_h - text_height) / 2
        draw.text((margin // 2, text_y_centered), cat_name_chinese, fill=(0,0,0), font=cat_label_f)

    # Calculate column width for splits

    placed_on_this_canvas_details = _draw_splits_on_canvas(
        draw, splits_to_render_on_this_canvas_now, 
        all_prepared_images_master,
        category_names_map, category_id_to_row_index, sorted_category_ids,
        margin, cat_width, cell_pad, img_spacing, 
        None, 
        short_length,
        cell_h, title_f,
        category_colors_map  # Pass category colors map
    )

    # Multi-category highlighting
    # HACK: 多类别标记 (细实线)
    # MULTI_CAT_BOX_COLOR = "purple"
    # MULTI_CAT_BOX_WIDTH = 1
    # if placed_on_this_canvas_details:
    #     logger.info(f"Drawing {len(placed_on_this_canvas_details)} multi-cat highlights for combined visualization.")
    #     for p_info in placed_on_this_canvas_details:
    #         if len(p_info['active_cats']) > 1:
    #             im_x,im_w = p_info['x'], p_info['w']
    #             for ac_id in p_info['active_cats']:
    #                 if ac_id not in category_id_to_row_index: continue
    #                 r_idx = category_id_to_row_index[ac_id]
    #                 hy0, hy1 = margin + r_idx*cell_h, margin+(r_idx+1)*cell_h
    #                 draw.rectangle((im_x,hy0,im_x+im_w,hy1),outline=MULTI_CAT_BOX_COLOR,width=MULTI_CAT_BOX_WIDTH)

    # Legend drawing
    lx,ly = canvas_width-cat_width-margin+20, margin
    for cid_legend in sorted_category_ids:
        draw.rectangle([lx,ly,lx+20,ly+20],fill=category_colors_map.get(cid_legend, (0,0,0)))
        draw.text((lx+30,ly),get_chinese_name(category_names_map.get(cid_legend, "N/A")),fill=(0,0,0),font=legend_f)
        ly+=30
    ly+=10
    # HACK: 多类别标记 (细实线)
    # draw.rectangle([lx,ly,lx+20,ly+20],outline=MULTI_CAT_BOX_COLOR,width=MULTI_CAT_BOX_WIDTH)
    # draw.text((lx+30,ly),"多类别标记 (细实线)",fill=(0,0,0),font=legend_f)

    # Log table for combined visualization
    _log_placed_images_table(placed_on_this_canvas_details, category_names_map, sorted_category_ids, splits_to_render_on_this_canvas_now)
    
    # Save combined visualization
    combined_output_path = output_path
    if single_image_per_split_mode:
        base_output_path = Path(output_path)
        combined_output_path = base_output_path.parent / f"{base_output_path.stem}_combined{base_output_path.suffix}"
    
    try:
        Path(combined_output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(combined_output_path, quality=95)
        logger.info(f"Successfully saved combined visualization:")
        logger.info(f"- Output path: {combined_output_path}")
        logger.info(f"- Image dimensions: {canvas_width}x{canvas_height}")
        logger.info(f"- Number of categories: {num_cat_rows}")
        logger.info(f"- Number of splits: {num_splits}")
        logger.info(f"- Total images placed: {len(placed_on_this_canvas_details)}")
    except Exception as e:
        logger.error(f"Error saving combined canvas to {combined_output_path}", exc_info=True)

    logger.info("Visualization process completed successfully.")

def create_square_visualization(coco_dir, output_path, short_length=70, rows=None):
    """Create a visualization of all images with annotations.
    
    Args:
        coco_dir (str): Path to COCO dataset directory
        output_path (str): Output path for visualization
        short_length (int): Shortest side length for scaled sub-images
        rows (int): Number of rows to arrange images. If None, will use square layout.
    """
    # Configure Loguru logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green>|L:<level>{level:<3}</level>| <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
    logger.add("logs/visualization_log.log", rotation="10 MB")
    
    logger.info(f"Starting visualization with parameters:")
    logger.info(f"- COCO directory: {coco_dir}")
    logger.info(f"- Output path: {output_path}")
    logger.info(f"- Short length: {short_length}")
    logger.info(f"- Number of rows: {rows}")

    # Load all images from all splits
    default_split_names = ['train', 'val', 'test']
    all_images = []
    skipped_log = []
    
    # Load category information from all splits
    category_names_map = {}
    for s_name in default_split_names:
        dataset = load_coco_dataset(os.path.join(coco_dir, 'annotations', f'instances_{s_name}2017.json'))
        if dataset and dataset.get('categories'):
            for cat in dataset['categories']:
                if cat['id'] not in category_names_map:
                    category_names_map[cat['id']] = cat['name']
    
    for s_name in default_split_names:
        dataset = load_coco_dataset(os.path.join(coco_dir, 'annotations', f'instances_{s_name}2017.json'))
        if not dataset:
            continue
            
        for img_info in dataset.get('images', []):
            img_path = os.path.join(coco_dir, f'{s_name}2017', img_info['file_name'])
            if not os.path.exists(img_path):
                skipped_log.append({'File': img_info['file_name'], 'Split': s_name, 'Reason': 'Image file not found'})
                continue
                
            try:
                orig_img = Image.open(img_path).convert('RGB')
                draw = ImageDraw.Draw(orig_img)
                try:
                    # Create different font sizes for filename and category labels
                    filename_font = ImageFont.truetype("Fonts/SimSun.ttf", 400)  # Larger font for filename
                    category_font = ImageFont.truetype("Fonts/SimSun.ttf", 200)  # Even larger font for category labels
                except IOError:
                    filename_font = ImageFont.load_default()
                    category_font = ImageFont.load_default()
                
                # Draw filename at the top of the image with background
                filename = img_info['file_name']
                text_bbox = draw.textbbox((0, 0), filename, font=filename_font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle([(0, 0), (text_width + 10, text_height + 10)], fill=(0, 0, 0))
                draw.text((5, 5), filename, fill=(255, 255, 255), font=filename_font)
                
                # Draw annotations
                current_img_annotations = [ann for ann in dataset.get('annotations', []) 
                                        if ann['image_id'] == img_info['id']]
                if current_img_annotations:
                    # Get unique colors for categories
                    category_colors = {}
                    for ann in current_img_annotations:
                        if ann['category_id'] not in category_colors:
                            category_colors[ann['category_id']] = CATEGORY_COLORS[len(category_colors) % len(CATEGORY_COLORS)]
                    
                    # Draw annotations
                    for ann in current_img_annotations:
                        bbox = ann['bbox']
                        x, y, w, h = bbox
                        color = category_colors[ann['category_id']]
                        
                        # Draw bounding box
                        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=10)
                        
                        # Draw category name
                        cat_name = get_chinese_name(category_names_map.get(ann['category_id'], 'Unknown'))
                        label = f"{cat_name}"
                        text_bbox = draw.textbbox((0, 0), label, font=category_font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # Calculate label position to be above the bounding box
                        padding = 5  # Padding between label and box
                        label_y = max(0, y - text_height - padding)  # Ensure label doesn't go above image
                        text_y = label_y + padding  # Add small padding for text
                        
                        # Ensure label doesn't go beyond right edge of image
                        label_width = min(text_width + 10, orig_img.width - x)
                        
                        # Draw label background above the bounding box
                        draw.rectangle([(x, label_y), (x + label_width, y)], fill=color)
                        draw.text((x + padding, text_y), label, fill=(255, 255, 255), font=category_font)
                
                all_images.append({
                    'image': orig_img,
                    'file_name': img_info['file_name'],
                    'width': orig_img.width,
                    'height': orig_img.height
                })
            except Exception as e:
                logger.warning(f"Error processing image {img_info['file_name']} from split {s_name}", exc_info=True)
                skipped_log.append({'File': img_info['file_name'], 'Split': s_name, 'Reason': f'Exception: {str(e)}'})
    
    if not all_images:
        logger.error("No valid images found to create visualization")
        return
        
    # Sort images by filename
    all_images.sort(key=lambda x: x['file_name'])
    
    # Calculate grid dimensions
    total_images = len(all_images)
    if rows is not None:
        # Fixed number of rows
        cols = int(np.ceil(total_images / rows))
        grid_size = (rows, cols)
    else:
        # Square-like layout
        grid_size = int(np.ceil(np.sqrt(total_images)))
        rows = grid_size
        cols = grid_size
    
    # Calculate scaled dimensions for all images
    for img_data in all_images:
        w, h = img_data['width'], img_data['height']
        if w < h:  # Portrait or square
            scale = short_length / w
            img_data['scaled_w'] = short_length
            img_data['scaled_h'] = int(h * scale)
        else:  # Landscape
            scale = short_length / h
            img_data['scaled_h'] = short_length
            img_data['scaled_w'] = int(w * scale)
    
    # Calculate canvas dimensions
    padding = 10
    max_row_width = max(sum(img['scaled_w'] for img in all_images[i:i+cols]) 
                       for i in range(0, total_images, cols))
    max_col_height = max(sum(img['scaled_h'] for img in all_images[i::cols]) 
                        for i in range(cols))
    
    canvas_width = max_row_width + padding * (cols + 1)
    canvas_height = max_col_height + padding * (rows + 1)
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    
    # Place images on canvas
    current_x = padding
    current_y = padding
    max_height_in_row = 0
    images_in_current_row = 0
    
    for i, img_data in enumerate(all_images):
        # Resize image
        resized_img = img_data['image'].resize(
            (img_data['scaled_w'], img_data['scaled_h']), 
            Image.Resampling.LANCZOS
        )
        
        # Paste image
        canvas.paste(resized_img, (current_x, current_y))
        
        # Update position for next image
        current_x += img_data['scaled_w'] + padding
        max_height_in_row = max(max_height_in_row, img_data['scaled_h'])
        images_in_current_row += 1
        
        # Move to next row if needed
        if images_in_current_row >= cols:
            current_x = padding
            current_y += max_height_in_row + padding
            max_height_in_row = 0
            images_in_current_row = 0
    
    # Save visualization
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)
        logger.info(f"Successfully saved visualization:")
        logger.info(f"- Output path: {output_path}")
        logger.info(f"- Image dimensions: {canvas_width}x{canvas_height}")
        logger.info(f"- Total images placed: {len(all_images)}")
        logger.info(f"- Layout: {rows} rows x {cols} columns")
        if skipped_log:
            sk_tbl = PrettyTable()
            sk_tbl.field_names = ["File", "Split", "Reason"]
            [sk_tbl.add_row(r.values()) for r in skipped_log]
            logger.info(f"Skipped Images:\n{sk_tbl.get_string()}")
    except Exception as e:
        logger.error(f"Error saving canvas to {output_path}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create COCO dataset visualization with dynamic canvas size based on content.')
    parser.add_argument('--coco_dir', type=str, required=True, help='Path to COCO dataset directory')
    parser.add_argument('--output', type=str, default='coco_visualization.jpg', help='Base output path for visualization(s)')
    parser.add_argument('--split_draw', type=str, nargs='+', default=None, help='Optional: Draw specific splits (e.g., train val) each to a separate image.')
    parser.add_argument('--short_length', type=int, default=70, help='Shortest side length for scaled sub-images on the canvas (default: 70).')
    parser.add_argument('--square', action='store_true', help='Create a square layout visualization of all images with annotations.')
    args = parser.parse_args()
    
    if args.square:
        create_square_visualization(args.coco_dir, args.output, short_length=args.short_length,rows=4)
    else:
        create_visualization(args.coco_dir, args.output, specified_splits_to_draw=args.split_draw, short_length=args.short_length) 