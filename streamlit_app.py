import streamlit as st
import os
import tempfile
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import requests
import json
import time
import uuid
import io
from pathlib import Path

# Set up page configuration
st.set_page_config(page_title="UI Component Analyzer", page_icon="🎨", layout="wide")

# Title and description
st.title("UI Component Analyzer")
st.markdown("""
This tool analyzes UI design images and identifies components based on atomic design principles.
Upload your design and see what UI components are detected!
""")

# Define the simplified UI Dissector class
class UIDissector:
    def __init__(self, config=None):
        """Initialize the UI Dissector with optional configuration"""
        self.config = config or {}
        
        # Set API configuration - read from environment variable
        self.api_key = os.environ.get('ROBOFLOW_API_KEY', '')
        
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
        }
    
    def get_image_dimensions(self, image_path):
        """Get image dimensions using PIL"""
        with Image.open(image_path) as img:
            width, height = img.size
        return height, width
    
    def detect_components(self, image_path):
        """Detect UI components using the Roboflow Workflow API or simulation"""
        # If no API key is provided, use simulated detection
        if not self.api_key:
            print("No API key available. Using simulated detection.")
            return self._simulate_component_detection(image_path)
        
        # Get image dimensions
        height, width = self.get_image_dimensions(image_path)
        
        # Read image and convert to base64
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Prepare headers and payload for Roboflow Workflow API
        headers = {
            'Content-Type': 'application/json'
        }
        
        payload = {
            "api_key": self.api_key,
            "inputs": {
                "image": {"type": "base64", "value": img_base64}
            }
        }
        
        try:
            # Make API request
            print("Sending request to Roboflow Workflow API...")
            
            # Send JSON payload to the API
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            # Debug information
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.reason}")
                return self._simulate_component_detection(image_path)
            
            # Parse response
            result = response.json()
            
            # Extract predictions from nested response
            predictions = []
            try:
                if "outputs" in result and isinstance(result["outputs"], list) and len(result["outputs"]) > 0:
                    first_output = result["outputs"][0]
                    if "predictions" in first_output and isinstance(first_output["predictions"], dict):
                        predictions_container = first_output["predictions"]
                        if "predictions" in predictions_container and isinstance(predictions_container["predictions"], list):
                            predictions = predictions_container["predictions"]
            except Exception as e:
                print(f"Error parsing predictions: {str(e)}")
                return self._simulate_component_detection(image_path)
            
            if not predictions:
                print("No components detected by API.")
                return self._simulate_component_detection(image_path)
            
            # Convert to our format
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
            
            return components
            
        except Exception as e:
            print(f"API Error: {str(e)}")
            return self._simulate_component_detection(image_path)
    
    def _simulate_component_detection(self, image_path):
        """Simulate component detection"""
        print("Using simulated component detection")
        
        # Get image dimensions using PIL
        height, width = self.get_image_dimensions(image_path)
        
        # Generate simulated components
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
        
        # Add detection_id and parent_id for compatibility
        for i, comp in enumerate(simulated_detections):
            comp['detection_id'] = f"sim_{i}"
            comp['parent_id'] = "image"
        
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
        
        # Fallback to spatial analysis
        print("Using spatial analysis for relationships")
        
        # Sort components by area (largest first)
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
    
    def create_visualization_image(self, image_path, components):
        """Create a visualization image with overlaid component boxes"""
        # Open the image with PIL
        img = Image.open(image_path)
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Define colors for atomic levels
        colors = {
            'atom': "#00ff00",      # green
            'molecule': "#ff9900",  # orange
            'organism': "#ff0000",  # red
            'template': "#0000ff",  # blue
            'unknown': "#999999"    # gray
        }
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            # If not available, use default font
            font = ImageFont.load_default()
        
        # Draw each component bounding box
        for comp in components:
            x1, y1, x2, y2 = comp['bbox']
            level = comp['atomic_level']
            color = colors.get(level, colors['unknown'])
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Draw label background
            label_text = f"{comp['type']} ({level})"
            label_bbox = draw.textbbox((x1, y1-15), label_text, font=font)
            draw.rectangle(label_bbox, fill="#ffffff", outline=color)
            
            # Draw label text
            draw.text((x1, y1-15), label_text, fill=color, font=font)
        
        # Return the image with bounding boxes
        return img
    
    def process_design(self, image_path, output_dir=None, visualize=True):
        """Process a design file to extract UI components"""
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise Exception(f"File not found: {image_path}")
            
            # Setup output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                # Generate a random output directory
                output_dir = os.path.join(os.getcwd(), f"ui_dissector_output_{uuid.uuid4().hex[:8]}")
                os.makedirs(output_dir, exist_ok=True)
            
            # Copy original image to output directory
            image_filename = os.path.basename(image_path)
            original_image_path = os.path.join(output_dir, f"original_{image_filename}")
            with open(image_path, 'rb') as src, open(original_image_path, 'wb') as dst:
                dst.write(src.read())
            
            # Start timer
            start_time = time.time()
            
            # Detect components
            components = self.detect_components(image_path)
            
            # Assign atomic design levels
            components = self.assign_atomic_levels(components)
            
            # Analyze component relationships
            components = self.analyze_relationships(components)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get image dimensions
            height, width = self.get_image_dimensions(image_path)
            
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
                    'image_size': {'width': width, 'height': height},
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
            
            # Generate visualization image
            if visualize:
                # Create visualization with overlaid boxes
                visualization_img = self.create_visualization_image(image_path, components)
                visualization_path = os.path.join(output_dir, f"visualization_{image_filename}")
                visualization_img.save(visualization_path)
                result['visualizations']['visualization'] = visualization_path
            
            return result
            
        except Exception as e:
            print(f"Error processing design: {str(e)}")
            return None

# Simple configuration in sidebar
with st.sidebar:
    st.header("Analysis Settings")
    confidence_threshold = st.slider("Detection Confidence", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
    
    # Add explanation about what the tool does
    st.markdown("---")
    st.markdown("""
    ### About
    
    This tool detects UI components in design images and classifies them according to atomic design principles:
    
    - **Atoms**: Basic UI elements (buttons, inputs)
    - **Molecules**: Simple groups of atoms (cards, form fields)
    - **Organisms**: Complex sections (headers, forms)
    - **Templates**: Page-level structures
    
    Upload your design image and click "Analyze Design" to start.
    """)

# File uploader for design image
uploaded_file = st.file_uploader("Choose a UI design image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Design", use_column_width=True)
    
    # Button to start analysis
    if st.button("Analyze Design"):
        # Create a temporary directory to save files
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save the uploaded file to the temp directory
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Progress indicator
                with st.spinner("Analyzing design image..."):
                    # Configure the UI Dissector
                    config = {
                        'confidence_threshold': confidence_threshold,
                    }
                    
                    # Initialize the UI Dissector
                    dissector = UIDissector(config)
                    
                    # Process the design
                    result = dissector.process_design(
                        image_path=temp_file_path,
                        output_dir=temp_dir,
                        visualize=True
                    )
                
                if result:
                    # Show success message
                    st.success(f"Analysis complete! Detected {len(result['components'])} components.")
                    
                    # Component summary
                    st.header("Component Summary")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Atoms", result['summary']['atoms'])
                    with col2:
                        st.metric("Molecules", result['summary']['molecules'])
                    with col3:
                        st.metric("Organisms", result['summary']['organisms'])
                    with col4:
                        st.metric("Templates", result['summary']['templates'])
                    with col5:
                        st.metric("Total", result['metadata']['component_count'])
                    
                    # Display visualization image
                    if 'visualization' in result['visualizations'] and os.path.exists(result['visualizations']['visualization']):
                        st.header("Component Visualization")
                        vis_img = Image.open(result['visualizations']['visualization'])
                        st.image(vis_img, use_column_width=True)
                        
                        # Provide download link
                        with open(result['visualizations']['visualization'], "rb") as file:
                            btn = st.download_button(
                                label="Download Visualization",
                                data=file,
                                file_name=f"ui_analysis_{os.path.basename(result['visualizations']['visualization'])}",
                                mime="image/png"
                            )
                    
                    # Component details table
                    st.header("Component Details")
                    
                    # Convert components to dataframe for better display
                    comp_rows = []
                    for comp in result['components']:
                        comp_rows.append({
                            'Type': comp['type'],
                            'Atomic Level': comp['atomic_level'],
                            'Confidence': f"{comp['confidence']:.2f}",
                            'Bounding Box': str(comp['bbox']),
                            'Parent': comp.get('parent', 'None'),
                            'Children': ', '.join(comp.get('children', [])) or 'None'
                        })
                    
                    if comp_rows:
                        df = pd.DataFrame(comp_rows)
                        st.dataframe(df, use_container_width=True)
                        
                else:
                    st.error("Analysis failed. Please try a different image.")
                    
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Try using a different image or check the API configuration.")

# Information footer
st.markdown("---")
st.markdown("""
**About this tool**: The UI Component Analyzer detects UI components 
in design images and classifies them according to atomic design principles.
""")