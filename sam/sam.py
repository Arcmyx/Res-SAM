from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import torch
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os

device = 'xpu'
torch.set_float32_matmul_precision('medium')  # Enable TF32 for faster computation


def find_boundary(vector):

    idx0 = np.argmax(vector != 0)
    
    if vector[idx0] == 0:
        return 0, 0

    idx1 = len(vector) - np.argmax(vector[::-1] != 0) - 1
    
    return idx0.item(), idx1.item()


def find_box(mask):
    vec0 = mask.sum(0)
    vec1 = mask.sum(1)
    
    x0, x1 = find_boundary(vec0)
    y0, y1 = find_boundary(vec1)
    return x0, y0, x1, y1


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


# sam = sam_model_registry["vit_h"](checkpoint="./sam_vit_h_4b8939.pth").to(device=device)
#sam = sam_model_registry["vit_l"](checkpoint=os.path.join(os.path.dirname(__file__), 'sam_vit_l_0b3195.pth')).to(device=device)
sam = sam_model_registry["vit_b"](checkpoint=os.path.join(os.path.dirname(__file__), 'sam_vit_b_01ec64.pth')).to(device=device)
#mask_generator = SamAutomaticMaskGenerator(
#    sam,
#    points_per_side=8,
#    pred_iou_thresh=0.95,
#    stability_score_thresh=0.95,
#    crop_n_layers=1,
#    box_nms_thresh=0.7,
 #   crop_n_points_downscale_factor=2,
#    min_mask_region_area=1000,
 #   output_mode='binary_mask'
#)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=4,          # Lower = fewer prompts (2x faster)
    pred_iou_thresh=0.9,        # Higher = fewer masks
    stability_score_thresh=0.9, # Higher = filter more
    crop_n_layers=0,            # Lower = fewer hierarchical crops (faster)
    crop_n_points_downscale_factor=4,  # Higher = fewer crop points
    min_mask_region_area=5000,  # Larger = skip small masks
    output_mode='binary_mask'
)
predictor = SamPredictor(sam)

# Cache for image embeddings
_cached_image = None
_cached_image_hash = None

def predict_mask(image):
    masks = mask_generator.generate(image)
    return [find_box(mask['segmentation'].astype(int)) for mask in masks if mask['area'] < 2e4]

def predict(image, input_point, input_label):
    import time
    global _cached_image, _cached_image_hash
    
    t0 = time.time()
    
    # Check if image changed by comparing hash
    image_hash = hash(image.tobytes())
    if image_hash != _cached_image_hash:
        predictor.set_image(image)
        _cached_image_hash = image_hash
        t1 = time.time()
        print(f"[SAM] set_image (NEW): {t1-t0:.3f}s")
    else:
        t1 = time.time()
        print(f"[SAM] set_image (CACHED): {t1-t0:.3f}s")
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    t2 = time.time()
    print(f"[SAM] predict: {t2-t1:.3f}s, total: {t2-t0:.3f}s")
    return masks[scores.argmax()]


if __name__ == '__main__':
    image_path = 'truck.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # predict_mask(image)

    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    mask = predict(image, input_point, input_label)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    show_box(find_box(mask), plt.gca())
    plt.show()
else:
    matplotlib.use("Agg")