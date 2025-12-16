import numpy as np
from PIL import Image
import colorsys

def generate_gpgp_data(width=640, height=360, num_blobs=15):
    # Create a blank array (low density)
    density_map = np.zeros((height, width), dtype=np.float32)

    # Generate random blobs
    for _ in range(num_blobs):
        # Random center
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        
        # Random sigma (spread)
        sigma = np.random.randint(20, 80)
        
        # Random intensity
        intensity = np.random.uniform(0.5, 1.0)
        
        # Create a grid of coordinates
        y, x = np.ogrid[:height, :width]
        
        # Gaussian function
        blob = intensity * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # Add to density map
        density_map += blob

    # Normalize to 0-1
    max_val = np.max(density_map)
    if max_val > 0:
        density_map = density_map / max_val
    
    # Save density map (Grayscale)
    density_img_array = (density_map * 255).astype(np.uint8)
    density_img = Image.fromarray(density_img_array, mode='L')
    density_img.save("gpgp_density.png")

    # Save heatmap (Colorized)
    # Simple heatmap: Blue (0) -> Green -> Red (1)
    heatmap_array = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            val = density_map[i, j]
            # Hue: 0.66 (Blue) -> 0.0 (Red)
            hue = (1.0 - val) * 0.66
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            heatmap_array[i, j] = (int(r * 255), int(g * 255), int(b * 255))
            
    heatmap_img = Image.fromarray(heatmap_array, mode='RGB')
    heatmap_img.save("gpgp_heatmap.png")
    
    # Create the obstacle map (ocean background)
    # Just a blue background
    ocean_array = np.zeros((height, width, 3), dtype=np.uint8)
    ocean_array[:] = (0, 100, 255) # RGB: Blueish
    ocean_img = Image.fromarray(ocean_array, mode='RGB')
    ocean_img.save("ocean.png")
    
    return density_map

if __name__ == "__main__":
    np.random.seed(42)
    generate_gpgp_data()
    print("Generated gpgp_density.png, gpgp_heatmap.png, and ocean.png using PIL")
