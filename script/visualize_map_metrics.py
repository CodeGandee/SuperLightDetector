import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import random
plt.switch_backend('agg')  # Switch to non-interactive backend

def create_detection_visualization(ground_truth, predictions, iou_threshold, title, output_path, metric_type):
    """
    Create a visualization showing ground truth and predictions with IoU threshold
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Set background color
    ax.set_facecolor('white')
    
    # Draw ground truth boxes in green
    for gt_box in ground_truth:
        x, y, w, h = gt_box
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth')
        ax.add_patch(rect)
    
    # Draw prediction boxes in red
    for pred_box in predictions:
        x, y, w, h = pred_box
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none', label='Prediction')
        ax.add_patch(rect)
    
    # Add title and legend
    ax.set_title(f'{title}', fontsize=16, pad=20)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Add color explanation and metric description
    text_x = 5
    text_y = 95
    
    ax.text(text_x, text_y, "Green: Ground Truth", fontsize=12, weight='bold')
    ax.text(text_x, text_y-10, "Red: Prediction", fontsize=12, weight='bold')
    
    # Add metric-specific explanation
    if metric_type == 'map50':
        ax.text(text_x, text_y-25, "mAP50: Detection is correct when IoU > 0.5", fontsize=12, weight='bold')
        ax.text(text_x, text_y-35, "• Most commonly used metric in object detection", fontsize=10)
        ax.text(text_x, text_y-40, "• Suitable for general object detection tasks", fontsize=10)
    elif metric_type == 'map75':
        ax.text(text_x, text_y-25, "mAP75: Detection is correct when IoU > 0.75", fontsize=12, weight='bold')
        ax.text(text_x, text_y-35, "• Stricter evaluation standard", fontsize=10)
        ax.text(text_x, text_y-40, "• Used when precise localization is crucial", fontsize=10)
    else:  # mAP
        ax.text(text_x, text_y-25, "mAP: Average precision over IoU range 0.5-0.95", fontsize=12, weight='bold')
        ax.text(text_x, text_y-35, "• Considers multiple IoU thresholds", fontsize=10)
        ax.text(text_x, text_y-40, "• Standard metric in COCO dataset evaluation", fontsize=10)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the visualization with tight layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def main():
    # mAP50 example - showing moderate overlap case
    ground_truth_map50 = [
        [20, 20, 30, 30]  # Single box case
    ]
    
    predictions_map50 = [
        [25, 25, 30, 30]  # Moderate overlap (IoU ≈ 0.6)
    ]
    
    # mAP75 example - showing high precision case
    ground_truth_map75 = [
        [20, 20, 30, 30]  # Single box case
    ]
    
    predictions_map75 = [
        [21, 21, 29, 29]  # High precision (IoU ≈ 0.85)
    ]
    
    # mAP example - showing perfect case
    ground_truth_map = [
        [20, 20, 30, 30]  # Single box case
    ]
    
    predictions_map = [
        [20, 20, 30, 30]  # Perfect match (IoU = 1.0)
    ]
    
    # Create visualization for mAP50
    create_detection_visualization(
        ground_truth_map50, 
        predictions_map50,
        0.5,
        'mAP50 Visualization',
        'map50_visualization.png',
        'map50'
    )
    
    # Create visualization for mAP75
    create_detection_visualization(
        ground_truth_map75,
        predictions_map75,
        0.75,
        'mAP75 Visualization',
        'map75_visualization.png',
        'map75'
    )
    
    # Create visualization for mAP
    create_detection_visualization(
        ground_truth_map,
        predictions_map,
        0.5,
        'mAP Visualization',
        'map_visualization.png',
        'map'
    )

if __name__ == '__main__':
    main()