import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import matplotlib.colors as mcolors
import torch.nn.functional as F

# VOC class names
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def get_random_color():
    colors = list(mcolors.CSS4_COLORS.values())
    return random.choice(colors)

def denormalize_image(img_tensor):
    img = img_tensor.cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    return np.clip(img, 0, 1)

@torch.no_grad()
def visualize_detr_output(model, dataloader, loss_fn, device, batch_size=4, score_thresh=0.7):
    images, targets, _ = next(iter(dataloader))
    random_idx = random.sample(range(len(images)), batch_size)
    images = torch.stack([images[i] for i in random_idx]).to(device)
    targets = [targets[i] for i in random_idx]

    model.eval()
    
    # Forward pass and get decoder outputs
    features = model.backbone(images)
    pos_enc = model.pos_enc(features)
    memory = model.encoder(features, pos_enc)
    query_pos = model.query_embed.weight.unsqueeze(0).repeat(images.size(0), 1, 1)
    tgt = torch.zeros_like(query_pos)
    hs, cross_attn_weights = model.decoder(tgt, memory, pos_enc, query_pos)
    pred_logits = model.class_mlp(hs)
    pred_bboxes = model.bbox_mlp(hs).sigmoid()
    
    # Post-processing
    output = loss_fn(pred_logits, pred_bboxes, targets, training=False, use_nms=True)
    detections = output['detections']
    
    image_h, image_w = images.shape[-2:]
    scale = torch.tensor([image_w, image_h, image_w, image_h], device=device)

    for det in detections:
        det['boxes'] *= scale

    # Visualization
    plt.figure(figsize=(18, 16))
    for i in range(batch_size):
        img = denormalize_image(images[i])
        
        # Original + Ground Truth
        plt.subplot(batch_size, 3, 3*i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Original + GT")
        for box, label in zip(targets[i]['boxes'], targets[i]['labels']):
            x1, y1, x2, y2 = box * scale.to('cpu')
            color = get_random_color()
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            plt.text(x1, y1, f'{VOC_CLASSES[label]}', color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))

        # Predictions
        plt.subplot(batch_size, 3, 3*i + 2)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Predicted Boxes")
        boxes, scores, labels = detections[i]['boxes'], detections[i]['scores'], detections[i]['labels']
        high_score_idx = [j for j, s in enumerate(scores) if s > score_thresh]

        selected_attn = []
        for j in high_score_idx:
            x1, y1, x2, y2 = boxes[j].cpu().numpy()
            color = get_random_color()
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            plt.text(x1, y1, f'{VOC_CLASSES[labels[j].item()]} {scores[j]:.2f}', color='white', fontsize=8, bbox=dict(facecolor=color, alpha=0.5))
            selected_attn.append(cross_attn_weights[-1][i][j])  # (N,)

        # Attention Map
        plt.subplot(batch_size, 3, 3*i + 3)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Attn Overlay")

        if selected_attn:
            attn_combined = torch.stack(selected_attn).mean(0)  # (N,)
            H_feat = image_h // 32
            W_feat = image_w // 32
            attn_map = attn_combined.reshape(H_feat, W_feat).unsqueeze(0).unsqueeze(0)
            attn_map = F.interpolate(attn_map, size=(image_h, image_w), mode='bilinear', align_corners=False)
            plt.imshow(attn_map.squeeze().cpu().numpy(), cmap='jet', alpha=0.5)

    plt.tight_layout()
    plt.show()
