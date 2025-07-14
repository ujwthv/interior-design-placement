import os
import base64
import io
import uuid
import numpy as np
import cv2
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import time
import urllib.error
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pathlib import Path
import gc
import wget
import json

# Configure PyTorch to reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Define constants
PLACES365_URL = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "resnet50_places365.pth.tar"
LABEL_FILE = "categories_places365.txt"
CATEGORY_LABELS_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"

# Umbrella categories
UMBRELLA_CATEGORIES = {
    "living room": {"living_room", "family_room", "lounge", "den", "television_room", "playroom", "recreation_room", "home_theater", "artists_loft", "attic"},
    "kitchen": {"kitchen", "dining_room", "pantry", "restaurant_kitchen", "cafeteria", "food_court"},
    "bathroom": {"bathroom", "shower", "jacuzzi/indoor", "sauna"},
    "bedroom": {"bedroom", "childs_room", "bedchamber", "dorm_room"},
    "open environment": {"mountain", "desert/sand", "desert/vegetation", "beach", "forest_path", "forest/broadleaf", "park", "field/cultivated", "field/wild", "field_road", "plaza", "highway", "viaduct", "mountain_path", "mountain_snowy", "rainforest", "lake/natural", "river", "valley", "vineyard", "wheat_field", "swamp", "glacier"}
}

# Object database
OBJECT_DATABASE = [
    {"id": "chair_001", "style": "modern", "type": "armchair", "suitable_scenes": ["living room", "bedroom"], "aesthetic_tags": ["sleek", "minimalist"], "unsuitable_scenes": ["open environment"], "group_id": None, "ref_size_pixels": (200, 300)},
    {"id": "chair_002", "style": "vintage", "type": "dining_chair", "suitable_scenes": ["kitchen", "living room"], "aesthetic_tags": ["ornate", "classic"], "unsuitable_scenes": ["bathroom"], "group_id": None, "ref_size_pixels": (180, 280)},
    {"id": "chair_003", "style": "modern", "type": "office_chair", "suitable_scenes": ["bedroom", "other indoor"], "aesthetic_tags": ["functional", "ergonomic"], "unsuitable_scenes": ["living room"], "group_id": None, "ref_size_pixels": (190, 290)},
    {"id": "chair_004", "style": "vintage", "type": "lounge_chair", "suitable_scenes": ["living room"], "aesthetic_tags": ["retro", "cozy"], "unsuitable_scenes": ["kitchen"], "group_id": None, "ref_size_pixels": (220, 320)},
    {"id": "cabinet_001", "style": "modern", "type": "cabinet", "suitable_scenes": ["kitchen", "living room"], "aesthetic_tags": ["sleek", "functional"], "unsuitable_scenes": ["open environment"], "group_id": None, "ref_size_pixels": (300, 400)},
]

STYLE_CLASSES = ["modern", "vintage"]
AESTHETIC_RULES = {
    "modern": {"allowed_tags": ["sleek", "minimalist", "functional", "ergonomic", "glass", "modular"], "disallowed_tags": ["ornate", "retro", "classic"]},
    "vintage": {"allowed_tags": ["ornate", "retro", "classic", "cozy", "wooden"], "disallowed_tags": ["sleek", "minimalist", "functional"]}
}

SCENE_TO_SURFACES = {
    "living room": ["floor", "rug", "table", "chair"],
    "kitchen": ["floor", "counter", "table", "cabinet"],
    "bedroom": ["floor", "bed", "rug", "bookshelf"],
    "bathroom": ["floor", "counter"],
    "open environment": ["grass", "dirt", "ground", "soil", "sand", "path", "patio"]
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SAM_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

USER_PREFERENCES = {}

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_scene_model():
    MODEL_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        st.write("Downloading Places365 model...")
        wget.download(PLACES365_URL, str(MODEL_PATH))
    scene_model = models.resnet50(num_classes=365)
    checkpoint = torch.load(MODEL_PATH, map_location=get_device())
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    scene_model.load_state_dict(state_dict)
    scene_model.eval()
    return scene_model.to(get_device())

def initialize_style_model():
    device = get_device()
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads = nn.Linear(model.heads.head.in_features, len(STYLE_CLASSES))
    model_path = "room_style_resnet50.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.write(f"Loaded trained model from {model_path}")
    else:
        st.warning(f"{model_path} not found. Using pre-trained weights without fine-tuning.")
    model.eval()
    return model.to(device)

def initialize_chair_style_model(weights_path='binary_chair_style_classifier.pth'):
    device = get_device()
    model = models.resnet50(weights='IMAGENET1K_V1')
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        st.warning(f"{weights_path} not found. Using default weights.")
    model.eval()
    return model.to(device)

def initialize_panoptic_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = get_device().type
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    return DefaultPredictor(cfg)

def initialize_sam_model():
    device = get_device()
    if not os.path.exists(SAM_CHECKPOINT):
        st.write("Downloading SAM checkpoint...")
        try:
            wget.download("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", SAM_CHECKPOINT)
        except Exception as e:
            st.error(f"Failed to download SAM checkpoint: {e}")
            raise
    try:
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=32,
            points_per_batch=64,
            pred_iou_thresh=0.75,
            stability_score_thresh=0.85,
            min_mask_region_area=50,
            box_nms_thresh=0.7
        )
        st.write("SAM model initialized successfully.")
        return mask_generator
    except Exception as e:
        st.error(f"Error initializing SAM model: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()

def initialize_depth_model():
    midas = None
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
            break
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                st.warning(f"Network error (Attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay)
            else:
                st.error(f"Failed to load MiDaS model after {max_retries} attempts: {e}")
                raise
    device = get_device()
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    return midas, midas_transforms.dpt_transform

def load_scene_labels():
    if not os.path.exists(LABEL_FILE):
        st.write("Downloading Places365 category labels...")
        wget.download(CATEGORY_LABELS_URL, LABEL_FILE)
    return [line.strip().split(' ')[0][3:] for line in open(LABEL_FILE)]

def classify_scene_label(scene_label, top_labels, top_probs):
    scene_base = scene_label.split('/')[-1]


    for umbrella, group in UMBRELLA_CATEGORIES.items():
        if scene_base in group:
            if umbrella == "bedroom":
                bedroom_labels = ["bedroom", "bedchamber", "childs_room", "dorm_room"]
                for label in bedroom_labels:
                    if label in top_labels and top_probs[0][top_labels.index(label)].item() > 0.1:
                        return "bedroom"
            return umbrella
    return "other indoor"

def extract_image_features(image, model):
    input_tensor = VAL_TRANSFORM(image).unsqueeze(0).to(get_device())
    with torch.no_grad():
        features = model(input_tensor)
    return features

def extract_color_texture_features(image):
    img = np.array(image)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    texture = cv2.Laplacian(img, cv2.CV_64F).var()
    features = np.concatenate([hist, [texture]])
    if len(features) < 2048:
        features = np.pad(features, (0, 2048 - len(features)), mode='constant')
    elif len(features) > 2048:
        features = features[:2048]
    return torch.tensor(features).unsqueeze(0).to(get_device())

def classify_style(image, model, confidence_threshold=0.5):
    input_tensor = VAL_TRANSFORM(image).unsqueeze(0).to(get_device())
    with torch.no_grad():
        logits = model(input_tensor)
        probs = nn.functional.softmax(logits, dim=1)
        top_prob, top_idx = probs.topk(1)
        predicted_style = STYLE_CLASSES[top_idx[0][0]]
        confidence = top_prob[0][0].item()
    if confidence < confidence_threshold:
        st.warning(f"Low confidence ({confidence:.4f}) for {predicted_style} classification. Falling back to 'modern'.")
        predicted_style = "modern"
    return predicted_style, confidence

def classify_chair_style(image, model, confidence_threshold=0.5):
    input_tensor = VAL_TRANSFORM(image).unsqueeze(0).to(get_device())
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = probs.topk(1)
        predicted_class = top_idx[0][0].item()
        confidence = top_prob[0][0].item()
    label = "modern" if predicted_class == 0 else "vintage"
    if confidence < confidence_threshold:
        st.warning(f"Low confidence ({confidence:.4f}) for {label} classification. Falling back to 'modern'.")
        label = "modern"
    return label, confidence

def infer_object_type(object_img, sam_mask_generator, chair_style_model):
    object_img_pil = object_img.resize((512, 512), Image.LANCZOS)
    object_rgb = np.array(object_img_pil)
    try:
        sam_masks = sam_mask_generator.generate(object_rgb)
        if not sam_masks:
            raise RuntimeError("No objects found with SAM.")
        suitable_masks = [m for m in sam_masks if 0.1 * 512 * 512 < m["area"] < 0.5 * 512 * 512]
        if not suitable_masks:
            raise RuntimeError("No suitable masks found.")
        best_mask = max(suitable_masks, key=lambda m: m["area"])
        mask = np.array(best_mask["segmentation"]).astype(np.uint8) * 255
    except Exception as e:
        st.error(f"Error in SAM mask generation: {e}")
        raise
    cropped_obj, cropped_mask, _ = crop_object(object_rgb, mask)
    cropped_obj_bgr = cv2.cvtColor(cropped_obj, cv2.COLOR_RGB2BGR)
    object_style, object_conf = classify_chair_style(object_img, chair_style_model)
    potential_types = [obj["type"] for obj in OBJECT_DATABASE if obj["style"] == object_style]
    object_type = potential_types[0] if potential_types else "armchair"
    return object_type, object_style, object_conf, cropped_obj_bgr, cropped_mask

def compute_cosine_similarity(features1, features2):
    device = features1.device
    cos = nn.CosineSimilarity(dim=1, eps=1e-8).to(device)
    similarity = cos(features1, features2)
    return similarity.item()

def check_aesthetic_compatibility(room_style, object_style, object_aesthetic_tags, style_model_trained):
    if not style_model_trained:
        return True, "Style model untrained; skipping aesthetic compatibility check."
    if room_style != object_style:
        return False, f"Style mismatch - Room: {room_style}, Object: {object_style}."
    rules = AESTHETIC_RULES.get(room_style, {})
    allowed = rules.get("allowed_tags", [])
    disallowed = rules.get("disallowed_tags", [])
    if not all(tag in allowed or tag not in disallowed for tag in object_aesthetic_tags):
        return False, f"Aesthetic tags {object_aesthetic_tags} incompatible with {room_style}."
    return True, ""

def check_semantic_compatibility(object_data, scene_category, object_type):
    if scene_category in object_data.get("unsuitable_scenes", []):
        return False, f"{object_type} unsuitable for {scene_category}."
    if scene_category not in object_data.get("suitable_scenes", []):
        return False, f"{object_type} not suitable for {scene_category}."
    return True, ""

def find_object_in_database(object_type, object_style):
    match = next((obj for obj in OBJECT_DATABASE if obj["type"] == object_type and obj["style"] == object_style), None)
    if match:
        return match
    fallback = next((obj for obj in OBJECT_DATABASE if obj["type"] == object_type), None)
    if fallback:
        st.warning(f"No {object_type} with style {object_style}. Using {fallback['style']} {object_type}.")
        return fallback
    st.warning(f"No {object_type} in database. Using default armchair.")
    return {
        "id": "chair_default", "style": "modern", "type": "armchair", "suitable_scenes": ["living room", "bedroom"],
        "aesthetic_tags": ["sleek", "minimalist"], "unsuitable_scenes": ["open environment"], "group_id": None,
        "ref_size_pixels": (200, 300)
    }

def compute_compatibility_score(room_features, object_features, room_style, object_style, scene_category, aesthetic_compatible, style_model_trained, object_type):
    device = room_features.device
    style_similarity = compute_cosine_similarity(room_features, object_features.to(device)) if style_model_trained else 0.5
    style_match_score = 1.0 if room_style == object_style else 0.0
    context_weight = 1.0 if scene_category in ["living room", "bedroom"] else 0.8
    suitable_scenes = next((obj["suitable_scenes"] for obj in OBJECT_DATABASE if obj["style"] == object_style), [])
    scene_suitability = 1.0 if scene_category in suitable_scenes else 0.3
    aesthetic_score = 1.0 if aesthetic_compatible else 0.0
    return 0.35 * style_similarity + 0.25 * style_match_score + 0.25 * scene_suitability * context_weight + 0.15 * aesthetic_score

def preprocess_image(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def compute_depth_map(room_img, midas, midas_transform):
    img_rgb = np.array(room_img.convert("RGB"))
    device = get_device()
    input_batch = midas_transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = F.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    midas = midas.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    return prediction.cpu().numpy()

def detect_and_segment_objects(room_img, scene_category):
    device = get_device()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = device.type
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    predictor = DefaultPredictor(cfg)
    room_img = preprocess_image(room_img)
    if len(room_img.shape) == 3 and room_img.shape[2] > 3:
        room_img = cv2.cvtColor(room_img, cv2.COLOR_BGR2RGB)[:, :, :3]
    elif len(room_img.shape) == 2:
        room_img = cv2.cvtColor(room_img, cv2.COLOR_GRAY2RGB)
    room_img = room_img.copy()
    outputs = predictor(room_img)
    instances = outputs["instances"].to(device)
    coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    coco_classes = coco_metadata.thing_classes
    detected_object_types = set()
    detected_indices = {}
    for i, class_idx in enumerate(instances.pred_classes):
        class_name = coco_classes[class_idx].lower()
        detected_object_types.add(class_name)
        if class_name not in detected_indices:
            detected_indices[class_name] = []
        detected_indices[class_name].append(i)
    target_instances = instances
    masks = target_instances.pred_masks.to(device)
    keep = []
    for i in range(len(masks)):
        overlap = False
        for j in keep:
            iou = (masks[i] & masks[j]).sum().float() / (masks[i] | masks[j]).sum().float()
            if iou > 0.5:
                overlap = True
                break
        if not overlap:
            keep.append(i)
    target_instances = target_instances[keep]
    placeable_labels = SCENE_TO_SURFACES.get(scene_category, ["floor"])
    placeable_mask = np.zeros_like(room_img[:, :, 0], dtype=np.uint8)
    if "panoptic_seg" in outputs:
        floor_detected = False
        for segment in outputs["panoptic_seg"][1]:
            category_id = segment["category_id"]
            try:
                category_name = coco_metadata.stuff_classes[category_id].lower()
                if any(label in category_name for label in placeable_labels) or "carpet" in category_name:
                    placeable_mask |= (outputs["panoptic_seg"][0].cpu().numpy() == segment["id"])
                    if "floor" in category_name or "rug" in category_name or "carpet" in category_name:
                        floor_detected = True
            except IndexError:
                continue
        if not floor_detected and scene_category != "open environment":
            depth_map = compute_depth_map(Image.fromarray(room_img[:, :, ::-1]), initialize_depth_model()[0], initialize_depth_model()[1])
            lower_third = room_img[int(2 * room_img.shape[0] / 3):, :]
            depth_lower = depth_map[int(2 * depth_map.shape[0] / 3):]
            floor_depth = np.median(depth_lower[depth_lower > 0])
            placeable_mask[depth_map < floor_depth * 1.5] = 1
    else:
        h, w = room_img.shape[:2]
        placeable_mask[int(h * 0.5):, :] = 1
    predictor = None
    gc.collect()
    torch.cuda.empty_cache()
    return target_instances.pred_masks, detected_indices, placeable_mask, outputs["panoptic_seg"][0].to(device), outputs["panoptic_seg"][1], list(detected_object_types)

def visualize_results(room_img, panoptic_seg, segments_info, placeable_mask, depth_map, scene_category, bounding_box=None, remove_bounding_box=None, indices=None, masks=None, detected_object_types=None):
    st.write("Generating visualization...")
    fig = plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(room_img[:, :, ::-1])
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 5, 2)
    v = Visualizer(room_img, MetadataCatalog.get("coco_2017_val_panoptic_separated"))
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    plt.imshow(out.get_image())
    plt.title("Panoptic Segmentation")
    plt.axis("off")
    plt.subplot(1, 5, 3)
    plt.imshow(room_img[:, :, ::-1])
    if indices and masks is not None and len(masks) > 0:
        for obj_type, idx_list in indices.items():
            for idx in idx_list:
                if idx < len(masks):
                    mask = masks[idx].cpu().numpy()
                    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    x, y, w, h = cv2.boundingRect(contours[0]) if contours else (0, 0, 10, 10)
                    plt.text(x, y - 10, f"{obj_type} {idx}", color='red', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
                    if remove_bounding_box and (x, y, w, h) == remove_bounding_box:
                        plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2))
    plt.title("Detected Objects with Indices")
    plt.axis("off")
    plt.subplot(1, 5, 4)
    white_mask = (placeable_mask == 1)
    if np.any(white_mask):
        plt.imshow(white_mask, cmap="gray")
        if bounding_box:
            x, y, w, h = bounding_box
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
            plt.text(x + w//2, y + h//2, "Placement Region", color='red', fontsize=12, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7))
    else:
        plt.imshow(np.zeros_like(placeable_mask, dtype=np.uint8), cmap="gray")
        if bounding_box:
            x, y, w, h = bounding_box
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
            plt.text(x + w//2, y + h//2, "Placement Region", color='red', fontsize=12, fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.title(f"Placeable Surfaces ({scene_category})")
    plt.axis("off")
    plt.subplot(1, 5, 5)
    plt.imshow(depth_map, cmap="inferno")
    plt.title("Depth Map")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches="tight", dpi=300)
    plt.close()
    gc.collect()
    return buf

def update_style_preference(room_style, object_style, user_accept):
    if room_style not in USER_PREFERENCES:
        USER_PREFERENCES[room_style] = {}
    USER_PREFERENCES[room_style][object_style] = USER_PREFERENCES[room_style].get(object_style, 0) + (1 if user_accept else -1)

def select_placement_region(placeable_mask, img_np):
    white_mask = (placeable_mask == 1)
    if not np.any(white_mask):
        st.warning("No placeable surfaces found. Using entire image as fallback.")
        return (0, 0, img_np.shape[1], img_np.shape[0])
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white_mask.astype(np.uint8), connectivity=8)
    min_area = 1000
    largest_area = 0
    largest_region = None
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            if stats[i, cv2.CC_STAT_AREA] > largest_area:
                largest_area = stats[i, cv2.CC_STAT_AREA]
                largest_region = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT]
                )
    if largest_region is None:
        st.warning("No suitable placeable regions found. Using largest region as fallback.")
        largest_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1 if stats.shape[0] > 1 else 0
        largest_region = (
            stats[largest_idx, cv2.CC_STAT_LEFT],
            stats[largest_idx, cv2.CC_STAT_TOP],
            stats[largest_idx, cv2.CC_STAT_WIDTH],
            stats[largest_idx, cv2.CC_STAT_HEIGHT]
        )
    return largest_region

def crop_object(object_img, mask):
    mask_binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No valid contours found in object mask.")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_obj = object_img[y:y+h, x:x+w].copy()
    cropped_mask = mask_binary[y:y+h, x:x+w].copy()
    mask_3d = np.repeat(cropped_mask[:, :, np.newaxis], 3, axis=2) // 255
    cropped_obj = np.where(mask_3d > 0, cropped_obj, 0).astype(np.uint8)
    return cropped_obj, cropped_mask, (x, y, w, h)

def place_object(room_img, obj_img, obj_mask, position, depth_map, scene_category, scale=1.0, angle=0, floor_level=None):
    x_place, y_place = position
    depth_value = depth_map[y_place, x_place]
    ref_depth = np.percentile(depth_map[depth_map > 0], 70)
    depth_scale = min(1.5, max(0.7, depth_value / ref_depth))
    h, w = obj_img.shape[:2]
    target_h = int(h * depth_scale * scale)
    target_w = int(w * depth_scale * scale)
    scaled_obj = cv2.resize(obj_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(obj_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if angle != 0:
        h, w = scaled_obj.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        scaled_obj = cv2.warpAffine(scaled_obj, M, (w, h))
        scaled_mask = cv2.warpAffine(scaled_mask, M, (w, h))
    obj_mask_binary = (scaled_mask > 0).astype(np.uint8)
    alpha = np.repeat(obj_mask_binary[..., None], 3, axis=2).astype(float)
    oh, ow = scaled_obj.shape[:2]
    if floor_level is not None:
        top_left_y = floor_level - oh
    else:
        top_left_y = max(0, y_place - oh // 2)
    top_left_x = max(0, x_place - ow // 2)
    top_left_x = min(top_left_x, room_img.shape[1] - ow)
    top_left_y = min(top_left_y, room_img.shape[0] - oh)
    actual_x = x_place - (ow // 2 - (top_left_x - max(0, x_place - ow // 2)))
    actual_y = y_place - (oh // 2 - (top_left_y - max(0, y_place - oh // 2)))
    roi = room_img[top_left_y:top_left_y + oh, top_left_x:top_left_x + ow]
    if roi.shape[:2] != scaled_obj.shape[:2]:
        scaled_obj = cv2.resize(scaled_obj, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    if scaled_obj.shape != roi.shape or alpha.shape != roi.shape:
        raise ValueError(f"Shape mismatch: scaled_obj {scaled_obj.shape}, roi {roi.shape}, alpha {alpha.shape}")
    output_img = room_img.copy()
    roi = output_img[top_left_y:top_left_y + oh, top_left_x:top_left_x + ow]
    blended = (alpha * scaled_obj + (1 - alpha) * roi).astype(np.uint8)
    roi[:] = blended
    return output_img, (actual_x, actual_y)

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def main():
    st.set_page_config(page_title="Interior Design Placement", layout="wide")
    st.title("Interior Design Object Placement")

    # Initialize session state
    if 'stage' not in st.session_state:
        st.session_state.stage = 'upload'
    if 'room_img' not in st.session_state:
        st.session_state.room_img = None
    if 'object_img' not in st.session_state:
        st.session_state.object_img = None
    if 'click_position' not in st.session_state:
        st.session_state.click_position = None
    if 'placeable_mask' not in st.session_state:
        st.session_state.placeable_mask = None
    if 'scene_category' not in st.session_state:
        st.session_state.scene_category = None
    if 'depth_map' not in st.session_state:
        st.session_state.depth_map = None
    if 'cropped_obj_bgr' not in st.session_state:
        st.session_state.cropped_obj_bgr = None
    if 'cropped_mask' not in st.session_state:
        st.session_state.cropped_mask = None
    if 'object_type' not in st.session_state:
        st.session_state.object_type = None
    if 'object_style' not in st.session_state:
        st.session_state.object_style = None
    if 'room_style' not in st.session_state:
        st.session_state.room_style = None
    if 'bounding_box' not in st.session_state:
        st.session_state.bounding_box = None
    if 'img_np' not in st.session_state:
        st.session_state.img_np = None
    if 'scale' not in st.session_state:
        st.session_state.scale = 1.0

    try:
        # Stage 1: Upload images
        if st.session_state.stage == 'upload':
            st.header("Upload Room and Object Images")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Room Image")
                room_uploader = st.file_uploader("Drag and drop room image here", type=['png', 'jpg', 'jpeg'], key="room_uploader")
                if room_uploader:
                    st.session_state.room_img = Image.open(room_uploader).convert('RGB')
                    st.image(st.session_state.room_img, caption="Uploaded Room Image", use_column_width=True)
            
            with col2:
                st.subheader("Object Image")
                object_uploader = st.file_uploader("Drag and drop object image here", type=['png', 'jpg', 'jpeg'], key="object_uploader")
                if object_uploader:
                    st.session_state.object_img = Image.open(object_uploader).convert('RGB')
                    st.image(st.session_state.object_img, caption="Uploaded Object Image", use_column_width=True)
            
            if st.session_state.room_img and st.session_state.object_img:
                if st.button("Proceed to Analysis"):
                    st.session_state.stage = 'analyze'

        # Stage 2: Analyze images
        elif st.session_state.stage == 'analyze':
            st.header("Scene Analysis")
            device = get_device()
            scene_model = initialize_scene_model()
            img_np = np.array(st.session_state.room_img)[:, :, ::-1].copy()
            st.session_state.img_np = img_np
            input_tensor = IMAGE_TRANSFORM(st.session_state.room_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = scene_model(input_tensor)
                probs = nn.functional.softmax(logits, dim=1)
                top_probs, top_idxs = probs.topk(5)
            scene_labels = load_scene_labels()
            top_labels = [scene_labels[top_idxs[0][i]] for i in range(top_probs.size(1))]
            st.subheader("Top Scene Classifications")
            for i in range(len(top_probs[0])):
                st.write(f"{top_labels[i]}: {top_probs[0][i].item():.4f}")
            predicted_scene = top_labels[0]
            scene_category = classify_scene_label(predicted_scene, top_labels, top_probs)
            if top_probs[0][0].item() < 0.3:
                st.warning(f"Low confidence ({top_probs[0][0].item():.4f}) for scene classification ({predicted_scene}).")
            st.session_state.scene_category = scene_category
            st.write(f"**Detected Scene:** {predicted_scene}")
            st.write(f"**Scene Category:** {scene_category}")

            scene_model = scene_model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            if scene_category == "open environment":
                st.warning("Open environment detected. Indoor furniture may be unsuitable.")
                if st.button("Proceed Anyway"):
                    st.session_state.stage = 'segment'
                elif st.button("Cancel"):
                    st.session_state.stage = 'upload'
                    st.session_state.room_img = None
                    st.session_state.object_img = None
                    st.experimental_rerun()
            else:
                st.session_state.stage = 'segment'

        # Stage 3: Segmentation and visualization
        elif st.session_state.stage == 'segment':
            st.header("Segmentation and Placement Region")
            panoptic_predictor = initialize_panoptic_model()
            midas, midas_transform = initialize_depth_model()
            masks, detected_indices, placeable_mask, panoptic_seg, segments_info, detected_object_types = detect_and_segment_objects(st.session_state.img_np, st.session_state.scene_category)
            st.session_state.placeable_mask = placeable_mask
            depth_map = compute_depth_map(st.session_state.room_img, midas, midas_transform)
            st.session_state.depth_map = depth_map
            bounding_box = select_placement_region(placeable_mask, st.session_state.img_np)
            st.session_state.bounding_box = bounding_box
            x, y, w, h = bounding_box
            st.write(f"**Selected Placement Region:** Center ({x + w//2}, {y + h//2}), Size ({w}, {h})")

            st.subheader("Detected Objects")
            if detected_object_types:
                for obj_type in detected_object_types:
                    indices = detected_indices.get(obj_type, [])
                    st.write(f"- {obj_type.capitalize()}: {len(indices)} detected (Indices: {indices})")
            else:
                st.write("No objects detected.")

            viz_buf = visualize_results(st.session_state.img_np, panoptic_seg, segments_info, placeable_mask, depth_map, st.session_state.scene_category,
                                       bounding_box=bounding_box, indices=detected_indices, masks=masks, detected_object_types=detected_object_types)
            st.image(viz_buf, caption="Scene Analysis Visualization", use_column_width=True)

            st.header("Remove Objects (Optional)")
            if detected_object_types:
                remove_object_type = st.selectbox("Select object type to remove", ['none'] + list(detected_object_types))
                if remove_object_type != 'none':
                    remove_indices = detected_indices.get(remove_object_type, [])
                    if remove_indices:
                        selected_remove_index = st.selectbox(f"Select index for {remove_object_type}", remove_indices)
                        if st.button("Remove Object"):
                            remove_mask = masks[selected_remove_index].cpu().numpy()
                            contours, _ = cv2.findContours((remove_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            remove_bounding_box = cv2.boundingRect(contours[0]) if contours else None
                            if remove_bounding_box:
                                st.session_state.placeable_mask[remove_mask] = 1
                                st.session_state.img_np = cv2.inpaint(st.session_state.img_np, (remove_mask * 255).astype(np.uint8), 3, cv2.INPAINT_NS)
                                st.write(f"Inpainting completed for {remove_object_type} at index {selected_remove_index}.")
                                # Re-run segmentation
                                masks, detected_indices, st.session_state.placeable_mask, panoptic_seg, segments_info, detected_object_types = detect_and_segment_objects(st.session_state.img_np, st.session_state.scene_category)
                                st.session_state.bounding_box = select_placement_region(st.session_state.placeable_mask, st.session_state.img_np)
                                x, y, w, h = st.session_state.bounding_box
                                st.write(f"**Updated Placement Region:** Center ({x + w//2}, {y + h//2}), Size ({w}, {h})")
                                viz_buf = visualize_results(st.session_state.img_np, panoptic_seg, segments_info, st.session_state.placeable_mask, depth_map, st.session_state.scene_category,
                                                          bounding_box=st.session_state.bounding_box, remove_bounding_box=remove_bounding_box, indices=detected_indices, masks=masks, detected_object_types=detected_object_types)
                                st.image(viz_buf, caption="Updated Scene Analysis", use_column_width=True)
            if st.button("Proceed to Object Analysis"):
                st.session_state.stage = 'object_analysis'

        # Stage 4: Object analysis
        elif st.session_state.stage == 'object_analysis':
            st.header("Object Analysis")
            style_model = initialize_style_model()
            chair_style_model = initialize_chair_style_model()
            sam_mask_generator = initialize_sam_model()
            object_type, object_style, object_conf, cropped_obj_bgr, cropped_mask = infer_object_type(st.session_state.object_img, sam_mask_generator, chair_style_model)
            st.session_state.object_type = object_type
            st.session_state.object_style = object_style
            st.session_state.cropped_obj_bgr = cropped_obj_bgr
            st.session_state.cropped_mask = cropped_mask
            room_style, room_conf = classify_style(st.session_state.room_img, style_model)
            st.session_state.room_style = room_style
            st.write(f"**Room Style:** {room_style} (Confidence: {room_conf:.4f})")
            st.write(f"**Object Style:** {object_style} (Confidence: {object_conf:.4f})")
            st.write(f"**Inferred Object Type:** {object_type}")

            style_compatible = (room_style == object_style)
            if not style_compatible:
                st.error(f"Incompatible styles: Room is {room_style}, Object is {object_style}.")
                if st.button("Proceed Anyway"):
                    st.session_state.stage = 'placement'
                elif st.button("Cancel"):
                    st.session_state.stage = 'upload'
                    st.session_state.room_img = None
                    st.session_state.object_img = None
                    st.experimental_rerun()
            else:
                st.session_state.stage = 'placement'

        # Stage 5: Placement
        elif st.session_state.stage == 'placement':
            st.header("Place Object")
            object_data = find_object_in_database(st.session_state.object_type, st.session_state.object_style)
            x, y, w, h = st.session_state.bounding_box
            default_x, default_y = x + w // 2, y + h // 2

            # Display placeable region for clicking
            placeable_img = np.zeros_like(st.session_state.img_np)
            placeable_img[st.session_state.placeable_mask == 1] = [255, 255, 255]
            placeable_pil = Image.fromarray(placeable_img[:, :, ::-1])
            placeable_base64 = image_to_base64(placeable_pil)

            # HTML and JavaScript for clickable canvas
            html_code = f"""
            <div>
                <canvas id="canvas" width="{st.session_state.img_np.shape[1]}" height="{st.session_state.img_np.shape[0]}" style="border:2px solid black;"></canvas>
                <p>Click on the white placeable region to select placement position.</p>
            </div>
            <script>
                var canvas = document.getElementById('canvas');
                var ctx = canvas.getContext('2d');
                var img = new Image();
                img.src = 'data:image/png;base64,{placeable_base64}';
                img.onload = function() {{
                    ctx.drawImage(img, 0, 0);
                }};

                canvas.addEventListener('click', function(event) {{
                    var rect = canvas.getBoundingClientRect();
                    var x = event.clientX - rect.left;
                    var y = event.clientY - rect.top;
                    var imgData = ctx.getImageData(x, y, 1, 1).data;
                    if (imgData[0] === 255 && imgData[1] === 255 && imgData[2] === 255) {{
                        var clickData = {{x: x, y: y}};
                        var input = document.createElement('input');
                        input.type = 'hidden';
                        input.name = 'click_position';
                        input.value = JSON.stringify(clickData);
                        document.body.appendChild(input);
                        document.getElementById('submit_click').click();
                    }} else {{
                        alert('Please click within the white placeable region.');
                    }}
                }});

                // Draw default position
                ctx.beginPath();
                ctx.arc({default_x}, {default_y}, 10, 0, 2 * Math.PI);
                ctx.fillStyle = 'red';
                ctx.fill();
                ctx.strokeStyle = 'black';
                ctx.stroke();
            </script>
            <form id="click_form">
                <input type="submit" id="submit_click" style="display:none;">
            </form>
            """
            st.components.v1.html(html_code, height=st.session_state.img_np.shape[0] + 50)

            # Handle click submission
            click_position = st.experimental_get_query_params().get('click_position', [None])[0]
            if click_position:
                click_data = json.loads(click_position)
                st.session_state.click_position = (int(click_data['x']), int(click_data['y']))
                st.experimental_set_query_params()

            # Object type selection for replacement
            detected_indices = detect_and_segment_objects(st.session_state.img_np, st.session_state.scene_category)[1]
            if st.session_state.object_type in detected_indices and len(detected_indices[st.session_state.object_type]) > 0:
                st.subheader("Replace Existing Object")
                indices = detected_indices[st.session_state.object_type]
                selected_index = st.selectbox(f"Select {st.session_state.object_type} to replace", [-1] + indices, format_func=lambda x: "New Placement" if x == -1 else f"Index {x}")
            else:
                selected_index = -1
                st.write(f"No {st.session_state.object_type}(s) detected. Proceeding with new placement.")

            # Inpaint if replacing
            if selected_index != -1:
                masks = detect_and_segment_objects(st.session_state.img_np, st.session_state.scene_category)[0]
                replacement_mask = masks[selected_index].cpu().numpy()
                st.session_state.placeable_mask[replacement_mask] = 1
                st.session_state.img_np = cv2.inpaint(st.session_state.img_np, (replacement_mask * 255).astype(np.uint8), 3, cv2.INPAINT_NS)

            # Placement scale
            placement_x = st.session_state.click_position[0] if st.session_state.click_position else default_x
            placement_y = st.session_state.click_position[1] if st.session_state.click_position else default_y
            depth_value = st.session_state.depth_map[placement_y, placement_x] if placement_y < st.session_state.depth_map.shape[0] and placement_x < st.session_state.depth_map.shape[1] else np.median(st.session_state.depth_map[st.session_state.depth_map > 0])
            normalized_depth = (depth_value - st.session_state.depth_map.min()) / (st.session_state.depth_map.max() - st.session_state.depth_map.min() + 1e-8)
            pixels_per_meter = 200 * (1 - normalized_depth) + 100
            perspective_correction = 1.0 + (normalized_depth * 0.2)
            ref_width_m, ref_height_m = 0.8, 1.0
            target_width = int(ref_width_m * pixels_per_meter * perspective_correction)
            target_height = int(ref_height_m * pixels_per_meter)
            target_scale = target_height / st.session_state.cropped_obj_bgr.shape[0]
            target_scale = max(0.3, min(target_scale, 1.0))
            st.session_state.scale = st.slider("Adjust Scale", 0.3, 1.0, target_scale, 0.1)

            # Preview placement
            preview_img = st.session_state.img_np.copy()
            top_left_x = max(0, placement_x - target_width // 2)
            top_left_y = max(0, (y + h) - target_height)
            bottom_right_x = min(st.session_state.img_np.shape[1], placement_x + target_width // 2)
            bottom_right_y = min(st.session_state.img_np.shape[0], (y + h))
            cv2.rectangle(preview_img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)
            cv2.putText(preview_img, f"Scale: {st.session_state.scale:.2f}", (int(top_left_x), int(top_left_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            st.image(preview_img[:, :, ::-1], caption="Placement Preview", use_column_width=True)

            if st.button("Place Object"):
                output_img, placement_pos = place_object(
                    st.session_state.img_np,
                    st.session_state.cropped_obj_bgr,
                    st.session_state.cropped_mask,
                    (placement_x, placement_y),
                    st.session_state.depth_map,
                    st.session_state.scene_category,
                    scale=st.session_state.scale,
                    floor_level=y + h
                )
                st.image(output_img[:, :, ::-1], caption=f"{st.session_state.object_type.capitalize()} Placed", use_column_width=True)
                st.write(f"**Placement Completed:** Position {placement_pos}, Scale: {st.session_state.scale:.2f}")

                # Compute compatibility
                style_model = initialize_style_model()
                room_features = torch.cat([extract_image_features(st.session_state.room_img, style_model), extract_color_texture_features(st.session_state.room_img)], dim=1)
                object_features = torch.cat([extract_image_features(st.session_state.object_img, style_model), extract_color_texture_features(st.session_state.object_img)], dim=1)
                aesthetic_compatible, aesthetic_issue = check_aesthetic_compatibility(st.session_state.room_style, st.session_state.object_style, object_data["aesthetic_tags"], True)
                semantic_compatible, semantic_issue = check_semantic_compatibility(object_data, st.session_state.scene_category, st.session_state.object_type)
                compatibility_score = compute_compatibility_score(room_features, object_features, st.session_state.room_style, st.session_state.object_style, st.session_state.scene_category, aesthetic_compatible, True, st.session_state.object_type)
                st.write(f"**Compatibility Score:** {compatibility_score:.2f}")
                if not (aesthetic_compatible and semantic_compatible and compatibility_score >= 0.7):
                    st.error("Incompatible placement: Aesthetic and semantic compatibility required with score >= 0.7.")
                else:
                    update_style_preference(st.session_state.room_style, st.session_state.object_style, True)

                if st.button("Start Over"):
                    st.session_state.stage = 'upload'
                    st.session_state.room_img = None
                    st.session_state.object_img = None
                    st.session_state.click_position = None
                    st.experimental_rerun()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        if st.button("Start Over"):
            st.session_state.stage = 'upload'
            st.session_state.room_img = None
            st.session_state.object_img = None
            st.session_state.click_position = None
            st.experimental_rerun()

if __name__ == "__main__":
    main()