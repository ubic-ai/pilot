import numpy as np
import cv2
from PIL import Image
import os
import json
import requests
import base64
from pathlib import Path
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import io
import uuid
import shutil

class UIDissector:
    def __init__(self, config=None):
        """Initialize the UI Dissector with optional configuration"""
        self.config = config or {}
        
        # Set API configuration
        self.api_key = self.config.get('api_key', os.environ.get('ROBOFLOW_API_KEY', ''))
        
        # Set Roboflow Workflow API endpoint
        self.api_url = self.config.get('api_url', 'https://serverless.roboflow.com/infer/workflows/ubic/custom-workflow')
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.4)
        
        # Dictionary mapping component types to atomic design levels
        self.atomic_mapping = {
            # Atoms - basic building blocks
            'button': 'atom',
            'checkbox': 'atom',
            'radio': 'atom',
            'toggle': 'atom',
            'icon': 'atom',
            'text': 'atom',
            'label': 'atom',
            'input': 'atom',
            'textbox': 'atom',
            'image': 'atom',
            'dropdown': 'atom',
            'link': 'atom',
            'slider': 'atom',
            
            # Molecules - simple combinations of atoms
            'form_field': 'molecule',
            'search_bar': 'molecule',
            'menu_item': 'molecule',
            'card': 'molecule',
            'pagination': 'molecule',
            'tab': 'molecule',
            
            # Organisms - complex UI sections
            'form': 'organism',
            'navigation': 'organism',
            'navbar': 'organism',
            'header': 'organism',
            'footer': 'organism',
            'sidebar': 'organism',
            'table': 'organism',
            
            # Templates - page-level structures
            'section': 'template',
            'container': 'template',
            'grid': 'template',
        }
        
        # Map API component names to our standardized names
        self.component_name_mapping = {
            'Button': 'button',
            'Input': 'input',
            'Textbox': 'textbox',
            'Checkbox': 'checkbox',
            'Radio': 'radio',
            'Dropdown': 'dropdown',
            'Toggle': 'toggle',
            'Slider': 'slider',
            'Icon': 'icon',
            'Image': 'image',
            'Text': 'text',
            'Label': 'label',
            'Link': 'link',
            'Card': 'card',
            'Form': 'form',
            'Navigation': 'navigation',
            'Navbar': 'navbar',
            'Header': 'header',
            'Footer': 'footer',
            'Sidebar': 'sidebar',
            'Section': 'section',
            'Container': 'container',
            # Add more mappings as needed based on API response
        }
        
        # Print configuration
        if not self.api_key:
            print("Warning: No API key provided. Using simulated detection.")
        else:
            print(f"UI Dissector initialized with API detection")
    
    def detect_components(self, image):
        """Detect UI components using the Roboflow Workflow API"""
        # If no API key is provided, use simulated detection
        if not self.api_key:
            print("No API key available. Using simulated detection.")
            return self._simulate_component_detection(image)
        
        # If image is a path, load it
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                img_bytes = image_file.read()
                
            # Also load with OpenCV to get dimensions
            img = cv2.imread(image)
            height, width = img.shape[:2]
        else:
            # Convert numpy array to bytes
            img = image.copy()
            height, width = img.shape[:2]
            _, img_bytes = cv2.imencode('.jpg', img)
            img_bytes = img_bytes.tobytes()
        
        # Convert image to base64 for JSON payload
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Prepare headers and payload according to the Roboflow Workflow API format
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Payload format for Roboflow Workflow API
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {"type": "base64", "value": img_base64}
            }
        }
        
        try:
            # Make API request
            print("Sending request to Roboflow Workflow API...")
            print(f"API URL: {self.api_url}")
            
            # For debugging, print a masked version of the API key
            if self.api_key:
                masked_key = self.api_key[:4] + '*' * (len(self.api_key) - 8) + self.api_key[-4:]
                print(f"Using API key: {masked_key}")
            
            # Send JSON payload to the API
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            # Debug information
            print(f"Response status code: {response.status_code}")
            
            # Handle different response status codes
            if response.status_code == 200:
                # Success
                print("API request successful")
            elif response.status_code == 401 or response.status_code == 403:
                print("Authentication error. Please check your API key.")
                print("Make sure you're using a valid Roboflow API key.")
                raise Exception(f"API authentication failed: {response.status_code} - {response.reason}")
            elif response.status_code == 429:
                print("Rate limit exceeded. Try again later.")
                raise Exception("API rate limit exceeded")
            else:
                print(f"Unexpected API response: {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"Error details: {error_details}")
                except:
                    print("Could not parse error details")
            
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Print the first 500 characters of the response for debugging
            response_str = json.dumps(result)
            print(f"Response preview: {response_str[:500]}{'...' if len(response_str) > 500 else ''}")
            
            # Parse the specific Roboflow Workflow API response structure
            # Based on the provided response structure:
            # {
            #   "outputs": [
            #     {
            #       "predictions": {
            #         "image": {
            #           "width": 2048,
            #           "height": 1536
            #         },
            #         "predictions": [
            #           {
            #             "width": 21.90625,
            #             "height": 15.515625,
            #             "x": 1488.0,
            #             "y": 393.5,
            #             "confidence": 0.9692381024360657,
            #             "class_id": 6,
            #             "class": "Icon",
            #             "detection_id": "045db2ce-4e41-4806-8fa3-9ed6d537a750",
            #             "parent_id": "image"
            #           },
            #           ...
            #         ]
            #       }
            #     }
            #   ]
            # }
            
            # Extract predictions
            predictions = []
            try:
                if "outputs" in result and isinstance(result["outputs"], list) and len(result["outputs"]) > 0:
                    first_output = result["outputs"][0]
                    if "predictions" in first_output and isinstance(first_output["predictions"], dict):
                        predictions_container = first_output["predictions"]
                        if "predictions" in predictions_container and isinstance(predictions_container["predictions"], list):
                            predictions = predictions_container["predictions"]
            except Exception as e:
                print(f"Error parsing predictions from response: {str(e)}")
                print("Using simulated detection as fallback")
                return self._simulate_component_detection(img)
            
            if not predictions:
                print("No predictions found in API response")
                print("Using simulated detection as fallback")
                return self._simulate_component_detection(img)
            
            print(f"Found {len(predictions)} predictions in API response")
            
            # Convert predictions to our format
            components = []
            for pred in predictions:
                try:
                    # Extract required fields
                    comp_class = pred.get("class", "unknown")
                    confidence = pred.get("confidence", 0.5)
                    x = pred.get("x", 0)
                    y = pred.get("y", 0)
                    w = pred.get("width", 0)
                    h = pred.get("height", 0)
                    
                    # Calculate bounding box in [x1, y1, x2, y2] format
                    # Note: The x,y in the API response are center coordinates
                    x1 = max(0, int(x - w/2))
                    y1 = max(0, int(y - h/2))
                    x2 = min(width, int(x + w/2))
                    y2 = min(height, int(y + h/2))
                    
                    # Map component type to our standardized name
                    component_type = self.component_name_mapping.get(
                        comp_class, comp_class.lower()
                    )
                    
                    components.append({
                        'type': component_type,
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2],
                        'detection_id': pred.get("detection_id", ""),
                        'parent_id': pred.get("parent_id", "")
                    })
                except Exception as e:
                    print(f"Error processing prediction: {str(e)}")
                    continue
            
            print(f"Processed {len(components)} components")
            
            # If no components were successfully processed, fall back to simulation
            if not components:
                print("No valid components could be processed from API response")
                print("Using simulated detection as fallback")
                return self._simulate_component_detection(img)
            
            return components
            
        except Exception as e:
            print(f"API Error: {str(e)}")
            print("Using simulated detection as fallback")
            return self._simulate_component_detection(img)
    
    def _simulate_component_detection(self, image):
        """Simulate component detection for demonstration purposes"""
        print("Using simulated component detection")
        
        # Get image dimensions
        if isinstance(image, str):
            img = cv2.imread(image)
            height, width = img.shape[:2]
        else:
            height, width = image.shape[:2]
        
        # Create a variety of simulated components based on image dimensions
        simulated_detections = [
            # Header at the top
            {'type': 'header', 'confidence': 0.98, 'bbox': [0, 0, width, int(height * 0.15)]},
            
            # Navigation below header
            {'type': 'navigation', 'confidence': 0.94, 'bbox': [0, int(height * 0.15), width, int(height * 0.25)]},
            
            # Main content area
            {'type': 'section', 'confidence': 0.92, 'bbox': [0, int(height * 0.25), width, int(height * 0.85)]},
            
            # Footer at bottom
            {'type': 'footer', 'confidence': 0.97, 'bbox': [0, int(height * 0.85), width, height]},
            
            # Form in main content
            {'type': 'form', 'confidence': 0.93, 'bbox': [int(width * 0.35), int(height * 0.35), int(width * 0.65), int(height * 0.65)]},
            
            # Buttons
            {'type': 'button', 'confidence': 0.95, 'bbox': [int(width * 0.8), int(height * 0.05), int(width * 0.9), int(height * 0.1)]},
            {'type': 'button', 'confidence': 0.94, 'bbox': [int(width * 0.4), int(height * 0.6), int(width * 0.5), int(height * 0.65)]},
            {'type': 'button', 'confidence': 0.93, 'bbox': [int(width * 0.55), int(height * 0.6), int(width * 0.65), int(height * 0.65)]},
            
            # Input fields
            {'type': 'input', 'confidence': 0.92, 'bbox': [int(width * 0.4), int(height * 0.4), int(width * 0.6), int(height * 0.45)]},
            {'type': 'input', 'confidence': 0.91, 'bbox': [int(width * 0.4), int(height * 0.5), int(width * 0.6), int(height * 0.55)]},
            
            # Checkbox
            {'type': 'checkbox', 'confidence': 0.89, 'bbox': [int(width * 0.4), int(height * 0.55), int(width * 0.42), int(height * 0.57)]},
            
            # Card in sidebar
            {'type': 'card', 'confidence': 0.88, 'bbox': [int(width * 0.1), int(height * 0.35), int(width * 0.3), int(height * 0.5)]},
            
            # Image in card
            {'type': 'image', 'confidence': 0.87, 'bbox': [int(width * 0.12), int(height * 0.37), int(width * 0.28), int(height * 0.43)]},
            
            # Text blocks
            {'type': 'text', 'confidence': 0.95, 'bbox': [int(width * 0.12), int(height * 0.44), int(width * 0.28), int(height * 0.48)]},
            {'type': 'text', 'confidence': 0.94, 'bbox': [int(width * 0.4), int(height * 0.3), int(width * 0.6), int(height * 0.35)]},
        ]
        
        return simulated_detections
    
    def assign_atomic_levels(self, components):
        """Assign atomic design levels to detected components"""
        for component in components:
            component['atomic_level'] = self.atomic_mapping.get(component['type'], 'unknown')
        return components
    
    def analyze_relationships(self, components):
        """Analyze containment and other relationships between components"""
        # Check if we have API-provided parent-child relationships
        has_api_relationships = all('detection_id' in comp and 'parent_id' in comp for comp in components)
        
        # If components have API-provided parent-child relationships
        if has_api_relationships:
            print("Using API-provided parent-child relationships")
            
            # Create lookup dictionary by detection_id
            comp_by_id = {comp.get('detection_id', ''): comp for comp in components if 'detection_id' in comp}
            
            # Initialize relationship data
            for comp in components:
                comp['children'] = []
                comp['parent'] = None
            
            # Analyze relationships based on parent_id
            for comp in components:
                parent_id = comp.get('parent_id', '')
                if parent_id and parent_id != 'image':  # 'image' is typically the root
                    # Find the parent component
                    parent_comp = comp_by_id.get(parent_id)
                    if parent_comp:
                        # Set parent-child relationship
                        comp['parent'] = parent_comp['type']
                        parent_comp['children'].append(comp['type'])
            
            return components
        
        # Fallback to spatial analysis if no API relationships
        print("No API relationships found, using spatial analysis")
        
        # Sort components by area (largest first) to ensure proper hierarchy
        components.sort(key=lambda c: (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]), reverse=True)
        
        # Initialize relationship data
        for comp in components:
            comp['children'] = []
            comp['parent'] = None
        
        # Analyze containment relationships
        for i, comp_a in enumerate(components):
            a_box = comp_a['bbox']
            
            for j, comp_b in enumerate(components):
                if i == j:
                    continue
                    
                b_box = comp_b['bbox']
                
                # Check if comp_b is contained within comp_a
                if (a_box[0] <= b_box[0] and a_box[1] <= b_box[1] and 
                    a_box[2] >= b_box[2] and a_box[3] >= b_box[3]):
                    
                    # Check if this is the closest container
                    if comp_b['parent'] is None:
                        comp_b['parent'] = comp_a['type']
                        comp_a['children'].append(comp_b['type'])
        
        return components
    
    def process_design(self, image_path, output_dir=None, visualize=True):
        """Process a design file to extract UI components and optionally generate visualizations
        
        Args:
            image_path: Path to the design image file
            output_dir: Directory to save results and visualizations (if None, uses temp directory)
            visualize: Whether to generate visualizations
            
        Returns:
            Dictionary containing analysis results and paths to visualizations
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise Exception(f"File not found: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Could not load image from {image_path}")
            
            # Setup output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                # Generate a random output directory in the current directory
                output_dir = os.path.join(os.getcwd(), f"ui_dissector_output_{uuid.uuid4().hex[:8]}")
                os.makedirs(output_dir, exist_ok=True)
            
            # Copy original image to output directory
            image_filename = os.path.basename(image_path)
            original_image_path = os.path.join(output_dir, f"original_{image_filename}")
            shutil.copy(image_path, original_image_path)
            
            # Start timer
            start_time = time.time()
            
            # Detect components
            components = self.detect_components(image)
            
            # Assign atomic design levels
            components = self.assign_atomic_levels(components)
            
            # Analyze component relationships
            components = self.analyze_relationships(components)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create structured output
            result = {
                'source_image': image_path,
                'components': components,
                'summary': {
                    'atoms': sum(1 for c in components if c['atomic_level'] == 'atom'),
                    'molecules': sum(1 for c in components if c['atomic_level'] == 'molecule'),
                    'organisms': sum(1 for c in components if c['atomic_level'] == 'organism'),
                    'templates': sum(1 for c in components if c['atomic_level'] == 'template'),
                    'unknown': sum(1 for c in components if c['atomic_level'] == 'unknown'),
                },
                'metadata': {
                    'image_size': {'width': image.shape[1], 'height': image.shape[0]},
                    'component_count': len(components),
                    'detection_method': 'api' if self.api_key else 'simulated',
                    'processing_time': processing_time,
                    'output_directory': output_dir,
                },
                'visualizations': {}
            }
            
            # Save results to output directory
            results_path = os.path.join(output_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(result, f, indent=2)
            
            result['metadata']['results_file'] = results_path
            
            # Generate visualizations if requested
            if visualize:
                visualization_paths = self._generate_visualizations(image, components, output_dir, image_filename)
                result['visualizations'] = visualization_paths
            
            return result
            
        except Exception as e:
            print(f"Error processing design: {str(e)}")
            return None
    
    def _generate_visualizations(self, image, components, output_dir, image_filename):
        """Generate visualizations of the detected components"""
        visualization_paths = {}
        
        # 1. Component detection visualization with bounding boxes
        detection_path = os.path.join(output_dir, f"detection_{image_filename}")
        self._visualize_detection(image, components, detection_path)
        visualization_paths['detection'] = detection_path
        
        # 2. Atomic design level visualization
        atomic_path = os.path.join(output_dir, f"atomic_levels_{image_filename}")
        self._visualize_atomic_levels(image, components, atomic_path)
        visualization_paths['atomic_levels'] = atomic_path
        
        # 3. Component hierarchy visualization
        hierarchy_path = os.path.join(output_dir, f"hierarchy_{image_filename}")
        self._visualize_hierarchy(image, components, hierarchy_path)
        visualization_paths['hierarchy'] = hierarchy_path
        
        # 4. Generate HTML report
        html_path = os.path.join(output_dir, "report.html")
        self._generate_html_report(image, components, html_path, visualization_paths)
        visualization_paths['html_report'] = html_path
        
        return visualization_paths
    
    def _visualize_detection(self, image, components, output_path):
        """Visualize detected components with bounding boxes"""
        # Create a copy of the image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # Draw bounding boxes for components
        for comp in components:
            x1, y1, x2, y2 = comp['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Generate a color based on component type
            # Use a hash of the component type for consistent colors
            color_seed = hash(comp['type']) % 10000 / 10000.0
            color = plt.cm.tab20(color_seed)
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height, 
                linewidth=2, 
                edgecolor=color,
                facecolor='none', 
                alpha=0.8
            )
            
            # Add the patch to the axis
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x1, y1-5, 
                f"{comp['type']} ({comp['confidence']:.2f})", 
                color=color, 
                fontsize=9, 
                weight='bold',
                alpha=0.9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
        
        # Add title
        ax.set_title("Detected UI Components")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def _visualize_atomic_levels(self, image, components, output_path):
        """Visualize components colored by atomic design level"""
        # Create a copy of the image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # Color mapping for atomic levels
        level_colors = {
            'atom': 'green',
            'molecule': 'orange',
            'organism': 'red',
            'template': 'blue',
            'unknown': 'gray'
        }
        
        # Draw bounding boxes for components
        for comp in components:
            x1, y1, x2, y2 = comp['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Get color based on atomic level
            color = level_colors.get(comp['atomic_level'], level_colors['unknown'])
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height, 
                linewidth=2, 
                edgecolor=color,
                facecolor=color, 
                alpha=0.2
            )
            
            # Add the patch to the axis
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x1, y1-5, 
                f"{comp['type']} ({comp['atomic_level']})", 
                color=color, 
                fontsize=9, 
                weight='bold',
                alpha=0.9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor=color, alpha=0.3, label=level.capitalize())
            for level, color in level_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        ax.set_title("Components by Atomic Design Level")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def _visualize_hierarchy(self, image, components, output_path):
        """Visualize component hierarchy relationships"""
        # Create a copy of the image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # Draw parent-child relationships
        for comp in components:
            if comp.get('children') and len(comp['children']) > 0:
                # Parent component
                p_x1, p_y1, p_x2, p_y2 = comp['bbox']
                parent_center = ((p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2)
                
                # Draw parent box
                parent_rect = patches.Rectangle(
                    (p_x1, p_y1), p_x2 - p_x1, p_y2 - p_y1, 
                    linewidth=3, 
                    edgecolor='blue',
                    facecolor='none', 
                    alpha=0.7
                )
                ax.add_patch(parent_rect)
                
                # Find children
                child_comps = []
                for c_type in comp['children']:
                    for c in components:
                        if c['type'] == c_type and c != comp:
                            child_comps.append(c)
                
                # Draw lines to children
                for child in child_comps:
                    c_x1, c_y1, c_x2, c_y2 = child['bbox']
                    child_center = ((c_x1 + c_x2) / 2, (c_y1 + c_y2) / 2)
                    
                    # Draw arrow from parent to child
                    ax.annotate(
                        "",
                        xy=child_center, xycoords='data',
                        xytext=parent_center, textcoords='data',
                        arrowprops=dict(
                            arrowstyle="-|>",
                            connectionstyle="arc3,rad=0.2",
                            color='blue',
                            alpha=0.6,
                            linewidth=1.5
                        )
                    )
                    
                    # Draw child box
                    child_rect = patches.Rectangle(
                        (c_x1, c_y1), c_x2 - c_x1, c_y2 - c_y1, 
                        linewidth=2, 
                        edgecolor='green',
                        facecolor='none', 
                        alpha=0.7
                    )
                    ax.add_patch(child_rect)
        
        # Add title
        ax.set_title("Component Hierarchy")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Tight layout
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
    
    def _generate_html_report(self, image, components, output_path, visualization_paths):
        """Generate an HTML report with all visualizations and component details"""
        # Convert image to base64 for embedding
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get component statistics
        atom_count = sum(1 for c in components if c['atomic_level'] == 'atom')
        molecule_count = sum(1 for c in components if c['atomic_level'] == 'molecule')
        organism_count = sum(1 for c in components if c['atomic_level'] == 'organism')
        template_count = sum(1 for c in components if c['atomic_level'] == 'template')
        
        # Group components by type
        component_types = {}
        for comp in components:
            comp_type = comp['type']
            if comp_type in component_types:
                component_types[comp_type] += 1
            else:
                component_types[comp_type] = 1
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>UI Dissector Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .report-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .stats {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 20px;
                }}
                .stat-card {{
                    background-color: #fff;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    flex: 1;
                    min-width: 120px;
                    text-align: center;
                }}
                .atom {{
                    border-left: 4px solid green;
                }}
                .molecule {{
                    border-left: 4px solid orange;
                }}
                .organism {{
                    border-left: 4px solid red;
                }}
                .template {{
                    border-left: 4px solid blue;
                }}
                .visualizations {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .visualization {{
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 10px;
                    border-radius: 5px;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                .component-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                .component-table th, .component-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .component-table th {{
                    background-color: #f2f2f2;
                }}
                .component-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .component-table tr:hover {{
                    background-color: #f1f1f1;
                }}
                footer {{
                    margin-top: 30px;
                    text-align: center;
                    font-size: 0.9em;
                    color: #777;
                    border-top: 1px solid #eee;
                    padding-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <div>
                    <h1>UI Component Analysis Report</h1>
                    <p>Analysis performed by UI Dissector</p>
                </div>
                <div>
                    <p><strong>Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
            
            <div class="summary-box">
                <h2>Analysis Summary</h2>
                <div class="stats">
                    <div class="stat-card atom">
                        <h3>{atom_count}</h3>
                        <p>Atoms</p>
                    </div>
                    <div class="stat-card molecule">
                        <h3>{molecule_count}</h3>
                        <p>Molecules</p>
                    </div>
                    <div class="stat-card organism">
                        <h3>{organism_count}</h3>
                        <p>Organisms</p>
                    </div>
                    <div class="stat-card template">
                        <h3>{template_count}</h3>
                        <p>Templates</p>
                    </div>
                    <div class="stat-card">
                        <h3>{len(components)}</h3>
                        <p>Total Components</p>
                    </div>
                </div>
                
                <h3>Component Types:</h3>
                <ul>
        """
        
        # Add component types
        for comp_type, count in component_types.items():
            html_content += f"<li><strong>{comp_type}:</strong> {count}</li>\n"
        
        html_content += """
                </ul>
            </div>
            
            <h2>Visualizations</h2>
            <div class="visualizations">
        """
        
        # Add visualizations
        for vis_name, vis_path in visualization_paths.items():
            if vis_name != 'html_report' and os.path.exists(vis_path):
                # Get the basename for display
                vis_basename = os.path.basename(vis_path)
                
                # Read and encode image
                with open(vis_path, "rb") as img_file:
                    vis_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                # Add visualization to HTML
                html_content += f"""
                <div class="visualization">
                    <h3>{vis_name.replace('_', ' ').title()}</h3>
                    <img src="data:image/png;base64,{vis_base64}" alt="{vis_name}">
                    <p>
                        <a href="{vis_basename}" target="_blank">Open in new window</a>
                    </p>
                </div>
                """
        
        html_content += """
            </div>
            
            <h2>Component Details</h2>
            <table class="component-table">
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Atomic Level</th>
                        <th>Confidence</th>
                        <th>Bounding Box</th>
                        <th>Parent</th>
                        <th>Children</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add component details
        for comp in components:
            bbox_str = ", ".join(str(b) for b in comp['bbox'])
            children_str = ", ".join(comp.get('children', []))
            parent_str = comp.get('parent', 'None')
            
            # Add IDs if available
            detection_id = comp.get('detection_id', '')
            parent_id = comp.get('parent_id', '')
            
            html_content += f"""
                    <tr>
                        <td>{comp['type']}</td>
                        <td>{comp['atomic_level']}</td>
                        <td>{comp['confidence']:.2f}</td>
                        <td>[{bbox_str}]</td>
                        <td>{parent_str}</td>
                        <td>{children_str}</td>
                    </tr>
            """
        
        html_content += """
                </tbody>
            </table>
            
            <footer>
                <p>Generated by UI Dissector - A tool for analyzing UI designs based on atomic design principles</p>
            </footer>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return output_path


if __name__ == "__main__":
    # Quick test
    # Get API key from environment variable
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    
    if not api_key:
        print("Warning: No ROBOFLOW_API_KEY environment variable found.")
        print("Set the environment variable or pass the API key in the config.")
        print("Using simulated detection for this test.")
    
    # Initialize dissector with API key from environment variable
    dissector = UIDissector({
        'api_key': api_key,
        'confidence_threshold': 0.4,
    })
    
    test_image_path = "test_design.png"
    
    if os.path.exists(test_image_path):
        result = dissector.process_design(test_image_path)
        if result:
            print("UI component analysis complete!")
            print(f"Results saved to: {result['metadata']['output_directory']}")
            print(f"HTML report: {result['visualizations']['html_report']}")
    else:
        print(f"Test image not found: {test_image_path}")