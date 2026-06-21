import torch
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from transformers import pipeline

# 1. Force CUDA Usage on your NVIDIA T1200
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} (Detected: {torch.cuda.get_device_name(0) if device=='cuda' else 'None'})")

# 2. Load the depth estimation model
print("Loading Depth Anything V2 (Small)...")
depth_estimator = pipeline(
    task="depth-estimation", 
    model="depth-anything/Depth-Anything-V2-Small-hf", 
    device=device
)

def create_3d_point_cloud_improved(image_path, downsample_factor=2, edge_threshold=0.03):
    """
    downsample_factor: Lower values = higher density (more points). Try 2 or 3 for sharper details.
    edge_threshold: Lower values = more aggressive curtain removal. Range: 0.01 to 0.1
    """
    raw_image = Image.open(image_path).convert("RGB")
    width, height = raw_image.size
    
    # Step 1: Run GPU inference to get Depth Map
    print("Estimating depth map locally via GPU...")
    predictions = depth_estimator(raw_image)
    depth_map = np.array(predictions["predicted_depth"])
    depth_map = np.array(Image.fromarray(depth_map).resize((width, height)))
    
    # Normalize depth maps to 0.0 - 1.0 range
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    rgb_encoded = np.array(raw_image)

    # Step 2: The Pinhole Camera Setup
    fx = fy = max(width, height)
    cx, cy = width / 2, height / 2
    
    x_coords, y_coords, z_coords = [], [], []
    colors = []
    
    # Step 3: Project with Edge-Aware Filtering
    print("Projecting to 3D and filtering depth discontinuities...")
    for v in range(downsample_factor, height - downsample_factor, downsample_factor):
        for u in range(downsample_factor, width - downsample_factor, downsample_factor):
            z = depth_map[v, u]
            if z == 0: continue
            
            # --- DISCONTINUITY FILTER ---
            # Look at horizontal and vertical neighbors to calculate local depth gradient
            depth_jump_horizontal = abs(z - depth_map[v, u - downsample_factor])
            depth_jump_vertical = abs(z - depth_map[v - downsample_factor, u])
            
            # If the depth changes too violently, it's a "curtain" pixel. Drop it!
            if depth_jump_horizontal > edge_threshold or depth_jump_vertical > edge_threshold:
                continue
            # -----------------------------
            
            # Compute physical 3D coordinates
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            
            x_coords.append(x)
            y_coords.append(-y) 
            z_coords.append(-z) 
            
            r, g, b = rgb_encoded[v, u]
            colors.append(f"rgb({r},{g},{b})")

    # Step 4: Render the filtered cloud
    print(f"Generating clean 3D space with {len(x_coords)} points...")
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=2, color=colors, opacity=1.0)
    )])
    
    fig.update_layout(
        title="Filtered 2D-to-3D Monocular Reconstruction (Curtain Artifacts Removed)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )
    fig.show()

if __name__ == "__main__":
    # Test it on the same image
    create_3d_point_cloud_improved("images/room_sample_2.jpg", downsample_factor=2, edge_threshold=0.03)