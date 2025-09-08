#!/usr/bin/env python3
"""
OBJ File Bounding Box Calculator

This script calculates the bounding box of a 3D object from an OBJ file.
It extracts all vertex coordinates and computes the minimum and maximum
values for each axis (X, Y, Z) to determine the object's bounding box.

Usage:
    python obj_bounding_box.py <path_to_obj_file>
    python obj_bounding_box.py --help
"""
import os
import sys
from typing import Tuple, List, Optional


def parse_obj_file(obj_file_path: str) -> List[Tuple[float, float, float]]:
    """
    Parse an OBJ file and extract all vertex coordinates.
    
    Args:
        obj_file_path: Path to the OBJ file
        
    Returns:
        List of (x, y, z) vertex coordinates
        
    Raises:
        FileNotFoundError: If the OBJ file doesn't exist
        ValueError: If the file format is invalid
    """
    if not os.path.exists(obj_file_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_file_path}")
    
    vertices = []
    
    try:
        with open(obj_file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse vertex coordinates (lines starting with 'v ')
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            z = float(parts[3])
                            vertices.append((x, y, z))
                        except ValueError as e:
                            print(f"Warning: Invalid vertex coordinates at line {line_num}: {line}")
                            continue
                    else:
                        print(f"Warning: Incomplete vertex data at line {line_num}: {line}")
                        continue
    
    except Exception as e:
        raise ValueError(f"Error reading OBJ file: {e}")
    
    if not vertices:
        raise ValueError("No valid vertices found in the OBJ file")
    
    return vertices


def calculate_bounding_box(vertices: List[Tuple[float, float, float]]) -> Tuple[
    Tuple[float, float, float],  # min_coords
    Tuple[float, float, float],  # max_coords
    Tuple[float, float, float]   # dimensions
]:
    """
    Calculate the bounding box from a list of vertices.
    
    Args:
        vertices: List of (x, y, z) vertex coordinates
        
    Returns:
        Tuple containing:
        - min_coords: (min_x, min_y, min_z)
        - max_coords: (max_x, max_y, max_z)
        - dimensions: (width, height, depth)
    """
    if not vertices:
        raise ValueError("No vertices provided")
    
    # Initialize min and max coordinates with the first vertex
    min_x = max_x = vertices[0][0]
    min_y = max_y = vertices[0][1]
    min_z = max_z = vertices[0][2]
    
    # Find min and max coordinates
    for x, y, z in vertices:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)
    
    min_coords = (min_x, min_y, min_z)
    max_coords = (max_x, max_y, max_z)
    dimensions = (max_x - min_x, max_y - min_y, max_z - min_z)
    
    return min_coords, max_coords, dimensions


def format_output(min_coords: Tuple[float, float, float],
                 max_coords: Tuple[float, float, float],
                 dimensions: Tuple[float, float, float],
                 vertex_count: int,
                 verbose: bool = False) -> str:
    """
    Format the bounding box information for output.
    
    Args:
        min_coords: Minimum coordinates (x, y, z)
        max_coords: Maximum coordinates (x, y, z)
        dimensions: Object dimensions (width, height, depth)
        vertex_count: Number of vertices processed
        verbose: Whether to include detailed information
        
    Returns:
        Formatted string with bounding box information
    """
    min_x, min_y, min_z = min_coords
    max_x, max_y, max_z = max_coords
    width, height, depth = dimensions
    
    output = []
    
    if verbose:
        output.append(f"OBJ File Analysis Results:")
        output.append(f"  Total vertices processed: {vertex_count}")
        output.append("")
    
    output.append("Bounding Box:")
    output.append(f"  Minimum coordinates: ({min_x:.6f}, {min_y:.6f}, {min_z:.6f})")
    output.append(f"  Maximum coordinates: ({max_x:.6f}, {max_y:.6f}, {max_z:.6f})")
    output.append("")
    output.append("Dimensions:")
    output.append(f"  Width (X):  {width:.6f}")
    output.append(f"  Height (Y): {height:.6f}")
    output.append(f"  Depth (Z):  {depth:.6f}")
    output.append("")
    output.append(f"  Volume: {width * height * depth:.6f}")
    
    return "\n".join(output)


def main():
    """Main function to handle command-line interface."""
    parser = argparse.ArgumentParser(
        description="Calculate the bounding box of a 3D object from an OBJ file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python obj_bounding_box.py model.obj
  python obj_bounding_box.py --verbose model.obj
  python obj_bounding_box.py --output bbox.txt model.obj
        """
    )
    
    parser.add_argument(
        "obj_file",
        help="Path to the OBJ file to analyze"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Include detailed information in output"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file to save results (default: print to stdout)"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse the OBJ file
        print(f"Parsing OBJ file: {args.obj_file}")
        vertices = parse_obj_file(args.obj_file)
        
        # Calculate bounding box
        min_coords, max_coords, dimensions = calculate_bounding_box(vertices)
        
        # Format output
        if args.format == "json":
            import json
            result = {
                "file": args.obj_file,
                "vertex_count": len(vertices),
                "bounding_box": {
                    "min_coords": list(min_coords),
                    "max_coords": list(max_coords),
                    "dimensions": list(dimensions),
                    "volume": dimensions[0] * dimensions[1] * dimensions[2]
                }
            }
            output_text = json.dumps(result, indent=2)
        else:
            output_text = format_output(
                min_coords, max_coords, dimensions, 
                len(vertices), args.verbose
            )
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"Results saved to: {args.output}")
        else:
            print(output_text)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
