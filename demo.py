#!/usr/bin/env python3
"""
UI Dissector Demo Script
This script demonstrates how to use the UI Dissector to analyze design files.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from dotenv import load_dotenv
from ui_dissector import UIDissector

# Load environment variables from .env file
load_dotenv()

# Component color mapping for visualization
COLORS = {
    'atom': 'green',
    'molecule': 'orange',
    'organism': 'red',
    'template': 'blue',
    'unknown': 'gray'
}

def visualize_components(image_path, components, output_path=None):
    """Visualize detected components with bounding boxes on the original image."""
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Draw bounding boxes for components
    for comp in components:
        x1, y1, x2, y2 = comp['bbox']
        width = x2 - x1
        height = y2 - y1
        
        # Get color based on atomic level
        color = COLORS.get(comp['atomic_level'], COLORS['unknown'])
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), width, height, 
            linewidth=2, 
            edgecolor=color, 
            facecolor='none',
            alpha=0.7
        )
        
        # Add the patch to the axis
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x1, y1-5, 
            f"{comp['type']} ({comp['atomic_level']})", 
            color=color, 
            fontsize=8, 
            weight='bold',
            alpha=0.9,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0)
        )
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='none', edgecolor=color, label=level)
        for level, color in COLORS.items()
        if level != 'unknown'
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    ax.set_title(f"UI Components - {os.path.basename(image_path)}")
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show the visualization
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def print_summary(result):
    """Print a summary of the analysis results."""
    print("\n===== UI DISSECTOR ANALYSIS SUMMARY =====")
    print(f"Source image: {os.path.basename(result['source_image'])}")
    print(f"Total components detected: {len(result['components'])}")
    print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")
    
    # Component count by atomic level
    print("\nComponent count by atomic design level:")
    for level, count in result['summary'].items():
        if level != 'total':
            print(f"  {level.capitalize()}: {count}")
    
    # Component types detected
    print("\nComponent types detected:")
    comp_types = {}
    for comp in result['components']:
        comp_type = comp['type']
        if comp_type in comp_types:
            comp_types[comp_type] += 1
        else:
            comp_types[comp_type] = 1
    
    for comp_type, count in comp_types.items():
        print(f"  {comp_type}: {count}")
    
    print("==========================================\n")

def main():
    parser = argparse.ArgumentParser(description='UI Dissector Demo')
    parser.add_argument('image_path', help='Path to design image')
    parser.add_argument('-o', '--output', help='Path to save results JSON')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--vis-output', help='Path to save visualization')
    parser.add_argument('-a', '--api-key', help='Roboflow API key (can also use ROBOFLOW_API_KEY env var)')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Confidence threshold (0.0-1.0)')
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    
    # Initialize UI Dissector
    dissector = UIDissector({
        'api_key': api_key,
        'confidence_threshold': args.threshold,
    })
    
    # Process design file
    print(f"Analyzing design file: {args.image_path}")
    result = dissector.process_design(args.image_path)
    
    if not result:
        print("Error: Failed to analyze design file")
        return
    
    # Print summary
    print_summary(result)
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Visualize results if requested
    if args.visualize or args.vis_output:
        visualize_components(args.image_path, result['components'], args.vis_output)

if __name__ == "__main__":
    main()