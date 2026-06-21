import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from transformers import pipeline
from rembg import remove  # NEW: Open-source segmentation backbone

# 1. Force CUDA Usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load the depth estimation model
print("Loading Depth Anything V2 (Small)...")
depth_estimator = pipeline(
    task="depth-estimation", 
    model="depth-anything/Depth-Anything-V2-Small-hf", 
    device=device
)

def create_3d_object_cloud(image_path, downsample_factor=2, edge_threshold=0.03):
    raw_image = Image.open(image_path).convert("RGB")
    width, height = raw_image.size
    
    # --- NEW: FOREGROUND SEGMENTATION ---
    print("Segmenting foreground object (U^2-Net via rembg)...")
    segmented_image = remove(raw_image)  # Returns RGBA image
    alpha_channel = np.array(segmented_image)[:, :, 3]  # Extract transparency mask
    # ------------------------------------
    
    # Step 1: Run GPU inference to get Depth Map
    print("Estimating depth map locally via GPU...")
    predictions = depth_estimator(raw_image)
    depth_map = np.array(predictions["predicted_depth"])
    depth_map = np.array(Image.fromarray(depth_map).resize((width, height)))
    
    # Normalize depth map
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    rgb_encoded = np.array(raw_image)

    # Step 2: Camera Intrinsics
    fx = fy = max(width, height)
    cx, cy = width / 2, height / 2
    
    x_coords, y_coords, z_coords = [], [], []
    colors = []
    
    print("Projecting isolated object to 3D...")
    for v in range(downsample_factor, height - downsample_factor, downsample_factor):
        for u in range(downsample_factor, width - downsample_factor, downsample_factor):
            
            # --- FOREGROUND MASK FILTER ---
            # If the pixel is part of the removed background (alpha < 128), skip it!
            if alpha_channel[v, u] < 128:
                continue
            # ------------------------------
            
            z = depth_map[v, u]
            if z == 0: continue
            
            # --- DISCONTINUITY FILTER ---
            depth_jump_horizontal = abs(z - depth_map[v, u - downsample_factor])
            depth_jump_vertical = abs(z - depth_map[v - downsample_factor, u])
            
            if depth_jump_horizontal > edge_threshold or depth_jump_vertical > edge_threshold:
                continue
            # -----------------------------
            
            # --- FLIP CORRECTION ---
            # Added a negative sign to X to fix the horizontal mirroring
            x = -(u - cx) * z / fx
            y = (v - cy) * z / fy
            
            x_coords.append(x)
            y_coords.append(-y) 
            z_coords.append(-z) 
            
            r, g, b = rgb_encoded[v, u]
            colors.append(f"rgb({r},{g},{b})")

    # Step 3: Render the isolated object
    print(f"Generating clean 3D object with {len(x_coords)} points...")
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=2, color=colors, opacity=1.0)
    )])
    
    fig.update_layout(
        title="Object-Centric 3D Reconstruction (Background Filtered)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    fig.show()

if __name__ == "__main__":
    create_3d_object_cloud("images/room_sample_2.jpg", downsample_factor=2, edge_threshold=0.03)