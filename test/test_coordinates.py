#!/usr/bin/env python3
"""
Test to locate precise known points and verify that projection work properly"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
import os

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from configs import CONFIG
from dataloading import load_bathy
from trajectory import pixel_to_latlon, latlon_to_pixel

# Fix the data directory path
CONFIG.data_dir = os.path.join(parent_dir, "data")


def test_location_localization(location_name, latitude, longitude, use_crop=True):
    """Generic test for localization of any location with optional crop around the point.
    
    Args:
        location_name (str): Name of the location to test
        latitude (float): Latitude of the location in degrees North
        longitude (float): Longitude of the location in degrees East (negative for West)
        use_crop (bool): If True, crop 100x100 pixels around the point for detailed view.
                        If False, show the full map.
    """
    
    print(f"🗺️  Test of {location_name} localization")
    print("="*50)
    
    print(f"GPS coordinates of {location_name}:")
    print(f"  Latitude:  {latitude:.6f}°N")
    print(f"  Longitude: {longitude:.6f}°E")
    print()
    
    # Load bathymetric data
    print("Loading bathymetric map...")
    ds = load_bathy(CONFIG)
    print(f"Map loaded: {ds.depth.shape} pixels, resolution ≈{ds.resolution_m:.1f} m/px")
    print()
    
    # Convert GPS coordinates to pixels
    try:
        location_col, location_row = latlon_to_pixel(ds.transform, ds.crs, latitude, longitude)
        print(f"Position on the map:")
        print(f"  Pixel (col, row): ({location_col:.2f}, {location_row:.2f})")
        
        # Check if the point is within map bounds
        H, W = ds.depth.shape
        if 0 <= location_col < W and 0 <= location_row < H:
            print(f"  ✅ Point within map bounds")
            depth_at_point = ds.depth[int(location_row), int(location_col)]
            print(f"  Depth at this point: {depth_at_point:.1f} m")
        else:
            print(f"  ❌ Point OUT OF map bounds ({W}×{H})")
            return
        
        # Test inverse conversion for verification
        back_lat, back_lon = pixel_to_latlon(ds.transform, ds.crs, location_col, location_row)
        err_lat = abs(back_lat - latitude)
        err_lon = abs(back_lon - longitude)
        print(f"  Inverse conversion verification:")
        print(f"    Back: {back_lat:.6f}°N, {back_lon:.6f}°E")
        print(f"    Error: {err_lat:.8f}° lat, {err_lon:.8f}° lon")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return
    
    print()
    if use_crop:
        print("📊 Creating map with crop around the point...")
        
        # Define crop area (100 pixels around the point)
        crop_size = 50  # 50 pixels on each side = 100x100 total

        # Calculate crop bounds
        col_start = max(0, int(location_col) - crop_size)
        col_end = min(W, int(location_col) + crop_size)
        row_start = max(0, int(location_row) - crop_size)
        row_end = min(H, int(location_row) + crop_size)
        
        # Extract cropped area
        depth_to_show = ds.depth[row_start:row_end, col_start:col_end]
        
        # Adjust point coordinates in the cropped system
        point_col_display = location_col - col_start
        point_row_display = location_row - row_start
        
        display_height, display_width = depth_to_show.shape
        print(f"  Crop: {display_width}×{display_height} pixels")
        print(f"  Position in crop: ({point_col_display:.2f}, {point_row_display:.2f})")
        
        # Figure settings for crop
        fig_size = (12, 12)
        title_suffix = f" (Crop {display_width}×{display_height} px)"
        extent = [0, display_width, display_height, 0]
        # Generate filename based on location name
        safe_name = location_name.lower().replace(' ', '_').replace("'", '').replace('.', '')
        output_filename = f"../{safe_name}_test_crop.png"
        
        # Add more frequent ticks for better precision in crop
        tick_step = max(1, min(10, display_width // 10))
    else:
        print("📊 Creating full map...")
        
        # Use full map
        depth_to_show = ds.depth
        point_col_display = location_col
        point_row_display = location_row
        display_height, display_width = depth_to_show.shape
        
        print(f"  Full map: {display_width}×{display_height} pixels")
        print(f"  Position on full map: ({point_col_display:.2f}, {point_row_display:.2f})")
        
        # Figure settings for full map
        fig_size = (15, 10)
        title_suffix = " (Full Map)"
        extent = [0, display_width, display_height, 0]
        # Generate filename based on location name
        safe_name = location_name.lower().replace(' ', '_').replace("'", '').replace('.', '')
        output_filename = f"../{safe_name}_test_full.png"
        
        # Less frequent ticks for full map
        tick_step = max(100, display_width // 20)
    
    print()
    
    # Create figure with bathymetric map
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Display bathymetry
    im = ax.imshow(depth_to_show, cmap='viridis', origin='upper', aspect='equal', 
                   vmin=-200, vmax=0, extent=extent)
    
    # Mark the location
    ax.plot(point_col_display, point_row_display, 'r*', markersize=25, 
            label=f'{location_name}\n({latitude:.4f}°N, {longitude:.4f}°E)', 
            markeredgecolor='white', markeredgewidth=3)
    
    # Display configuration
    ax.set_title(f'Bathymetric Map - {location_name}{title_suffix}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Pixels (East →)', fontsize=12)
    ax.set_ylabel('Pixels (South ↓)', fontsize=12)
    
    # Color bar for depth
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Depth (m)', fontsize=12)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, alpha=0.4, linewidth=0.5)
    
    # Add ticks based on crop/full map mode
    if use_crop:
        ax.set_xticks(np.arange(0, display_width+1, tick_step))
        ax.set_yticks(np.arange(0, display_height+1, tick_step))
    else:
        ax.set_xticks(np.arange(0, display_width+1, tick_step))
        ax.set_yticks(np.arange(0, display_height+1, tick_step))
    
    # Save the map
    output_path = Path(output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    map_type = "cropped" if use_crop else "full"
    print(f"🎯 {map_type.capitalize()} map saved: {output_path.absolute()}")
    print()
    print("✅ Test completed! Visually verify that the red star")
    print("   corresponds correctly to the point of interest.")
    if use_crop:
        print("   The map is cropped for better precision.")
    else:
        print("   The full map is shown for context.")
    print()
    
    # Information about the displayed area
    print(f"ℹ️  Information about the {map_type} area:")
    print(f"   • Displayed area: {display_width}×{display_height} pixels")
    print(f"   • Coverage: approximately {display_width * ds.resolution_m / 1000:.2f} × {display_height * ds.resolution_m / 1000:.2f} km")
    print(f"   • Resolution: {ds.resolution_m:.1f} m per pixel")
    print(f"   • Coordinate system: {ds.crs}")
    
    # Calculate depth statistics for the displayed area
    if use_crop and 'col_start' in locals() and 'row_start' in locals():
        mask_section = ds.mask[row_start:row_end, col_start:col_end]
    else:
        mask_section = ds.mask
    
    if np.any(mask_section):
        valid_depths = depth_to_show[mask_section]
        print(f"   • Min/max depth in {map_type}: {valid_depths.min():.1f} / {valid_depths.max():.1f} m")


if __name__ == "__main__":
    # Define location coordinates
    #1 Chateaubriand's Tomb (Grand-Bé Island, Saint-Malo)
    name = "Chateaubriand's Tomb"
    lat = 48.653031965489625  # Latitude North
    lon = -2.033150547088596  # Longitude West

    print("=== Testing with CROP disabled (full map) ===")
    test_location_localization(name, lat, lon, use_crop=False)
    print("=== Testing with CROP enabled ===")
    test_location_localization(name, lat, lon, use_crop=True)
    
    


