import easyocr
import cv2
import numpy as np
import json

def convert_annotations(json_annotations):
    """
    Convert JSON annotation format to coordinate tuples.
    
    Args:
        json_annotations (list): List of annotation dictionaries in JSON format
    Returns:
        list: List of annotation dictionaries with simplified coordinates
    """
    converted_annotations = []
    
    for ann in json_annotations:
        if ann['contentType'] == 'rect':
            rect = ann['rectMask']
            # Convert to (x1, y1, x2, y2) format
            coords = (
                int(rect['xMin']),
                int(rect['yMin']),
                int(rect['xMin'] + rect['width']),
                int(rect['yMin'] + rect['height'])
            )
            
            converted_annotations.append({
                'coords': coords,
                'label': ann['labels']['labelName']
            })
    
    return converted_annotations

def extract_text_from_annotations(image_path, json_annotations):
    """
    Extract text from specified JSON annotations.
    
    Args:
        image_path (str): Path to the image file
        json_annotations (list): List of annotation dictionaries in JSON format
    Returns:
        list: List of dictionaries with coordinates and detected text
    """
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Convert JSON annotations to coordinate format
    annotations = convert_annotations(json_annotations)
    
    results = []
    
    # Process each annotation
    for i, ann in enumerate(annotations):
        x1, y1, x2, y2 = ann['coords']
        
        # Ensure coordinates are within image boundaries
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        
        # Crop image to annotation coordinates
        cropped = image[y1:y2, x1:x2]
        
        # Skip if cropped image is empty
        if cropped.size == 0:
            continue
        
        # Detect text in the cropped region
        detections = reader.readtext(cropped)
        
        # Extract text and confidence
        texts = []
        for _, text, confidence in detections:
            texts.append({
                'text': text,
                'confidence': confidence
            })
        
        # Store results
        results.append({
            'annotation_id': i,
            'label': ann['label'],
            'coordinates': (x1, y1, x2, y2),
            'detected_text': texts
        })
    
    return results

def display_results(image_path, results):
    """
    Display image with detected text and bounding boxes
    """
    image = cv2.imread(image_path)
    
    for result in results:
        x1, y1, x2, y2 = result['coordinates']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add detected text and label
        y_text = y1 - 10
        cv2.putText(image, f"Label: {result['label']}", (x1, y_text),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        y_text -= 20
        
        for text_obj in result['detected_text']:
            text = f"{text_obj['text']} ({text_obj['confidence']:.2f})"
            cv2.putText(image, text, (x1, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_text -= 20
    
    # Save annotated image
    output_path = "annotated_" + image_path.split('/')[-1]
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Example usage
    image_path = "sample2.jpg"  # Replace with your image path
    
    # Load JSON annotations
    annotations = [
        {
            "content": [
                {"x": 690.8475566305694, "y": 588.2369902462316},
                {"x": 801.0916003866994, "y": 588.2369902462316},
                {"x": 801.0916003866994, "y": 692.3563649047987},
                {"x": 690.8475566305694, "y": 692.3563649047987}
            ],
            "rectMask": {
                "xMin": 690.8475566305694,
                "yMin": 588.2369902462316,
                "width": 110.24404375612993,
                "height": 104.11937465856715
            },
            "labels": {
                "labelName": "unnamed",
                "labelColor": "red",
                "labelColorRGB": "255,0,0",
                "visibility": False
            },
            "labelLocation": {
                "x": 745.9695785086344,
                "y": 640.2966775755151
            },
            "contentType": "rect"
        }
        # Add more annotations as needed
    ]
    
    try:
        # Extract text from annotations
        results = extract_text_from_annotations(image_path, annotations)
        
        # Display results
        print("\nDetected Text in Annotations:")
        for result in results:
            print(f"\nAnnotation {result['annotation_id']} (Label: {result['label']}):")
            print(f"Coordinates: {result['coordinates']}")
            print("Detected text:")
            for text_obj in result['detected_text']:
                print(f"- {text_obj['text']} (Confidence: {text_obj['confidence']:.2f})")
        
        # Visualize results
        display_results(image_path, results)
        print(f"\nAnnotated image saved as: annotated_{image_path.split('/')[-1]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")