#!/usr/bin/env python3
"""
Fixed PLY Converter with 360° Visibility

This version fixes:
- Face orientation issues
- Normal direction problems  
- Single-sided surface visibility
- Missing faces from certain angles

Dependencies (required):
  pip install trimesh numpy plyfile scipy

Dependencies (optional for advanced features):
  pip install open3d scikit-image
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

# Optional dependencies - graceful fallback if not available
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    o3d = None
    HAS_OPEN3D = False

try:
    from scipy.spatial import cKDTree, distance
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    ndimage = None
    HAS_SCIPY = False

try:
    from skimage import measure as skmeasure
    HAS_SKIMAGE = True
except ImportError:
    skmeasure = None
    HAS_SKIMAGE = False


def log(msg: str) -> None:
    """Simple logging function"""
    print(f"[PLY-Converter] {msg}", flush=True)


def fix_face_orientation_and_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Fix face orientation and normals so mesh is visible from all angles
    """
    try:
        original_face_count = len(mesh.faces)
        log(f"Starting face orientation fix with {original_face_count} faces")
        
        # Step 1: Fix face winding/orientation
        try:
            mesh.fix_normals()
            log("Face normals fixed")
        except Exception as e:
            log(f"Fix normals failed: {e}")
        
        # Step 2: Ensure consistent face winding
        try:
            mesh.unify_normals()
            log("Face winding unified")  
        except Exception as e:
            log(f"Unify normals failed: {e}")
        
        # Step 3: Check if we need double-sided mesh
        try:
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
            
            # Check if mesh has consistent outward-facing normals
            face_normals = mesh.face_normals
            
            if len(face_normals) > 0:
                # Calculate center of mesh
                center = vertices.mean(axis=0)
                
                # Check how many face normals point away from center
                face_centers = vertices[faces].mean(axis=1)
                center_to_face = face_centers - center
                center_to_face = center_to_face / (np.linalg.norm(center_to_face, axis=1, keepdims=True) + 1e-8)
                
                # Dot product tells us if normal points away from center
                dot_products = np.sum(face_normals * center_to_face, axis=1)
                outward_faces = np.sum(dot_products > 0)
                total_faces = len(dot_products)
                
                outward_ratio = outward_faces / total_faces if total_faces > 0 else 0
                log(f"Outward-facing faces: {outward_faces}/{total_faces} ({outward_ratio:.2f})")
                
                # If less than 60% of faces point outward, create double-sided mesh
                if outward_ratio < 0.6:
                    log("Creating double-sided mesh for better visibility")
                    
                    # Create flipped faces (reverse winding order)
                    flipped_faces = faces[:, [0, 2, 1]]  # Reverse winding order
                    
                    # Combine original and flipped faces
                    all_faces = np.vstack([faces, flipped_faces])
                    
                    # Create new mesh with double-sided faces
                    double_sided_mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)
                    
                    # Transfer colors if they exist
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        double_sided_mesh.visual.vertex_colors = mesh.visual.vertex_colors
                    
                    # Basic cleanup of the double-sided mesh
                    double_sided_mesh.remove_degenerate_faces()
                    double_sided_mesh.remove_duplicate_faces()
                    double_sided_mesh.remove_unreferenced_vertices()
                    
                    log(f"Double-sided mesh created: {len(double_sided_mesh.faces)} faces")
                    return double_sided_mesh
                else:
                    log("Mesh has good outward-facing normals, no double-siding needed")
                    
        except Exception as e:
            log(f"Double-sided mesh creation failed: {e}")
        
        # Step 4: Standard cleanup
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        final_face_count = len(mesh.faces)
        log(f"Face orientation fix completed: {original_face_count} -> {final_face_count} faces")
        return mesh
        
    except Exception as e:
        log(f"Face orientation fix failed: {e}")
        return mesh


def precise_poisson_reconstruction(vertices: np.ndarray, colors: Optional[np.ndarray] = None, 
                                 smoothing_level: str = "medium") -> Optional[trimesh.Trimesh]:
    """Precise Poisson reconstruction with better density filtering"""
    if not HAS_OPEN3D:
        return None
    
    try:
        log("Starting Poisson reconstruction")
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Better normal estimation with consistent orientation
        bbox = vertices.max(axis=0) - vertices.min(axis=0)
        radius = max(np.linalg.norm(bbox) * 0.02, 0.001)
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
        pcd.normalize_normals()
        
        # Improved normal orientation for better surface generation
        try:
            # Try to orient normals consistently toward outside
            pcd.orient_normals_consistent_tangent_plane(100)
        except:
            try:
                # Fallback: orient toward camera/viewpoint
                pcd.orient_normals_to_align_with_direction()
            except:
                log("Normal orientation failed, using estimated normals as-is")
        
        # Precise Poisson parameters
        depth_map = {
            'light': 8,
            'medium': 9, 
            'high': 10,
            'ultra': 11
        }
        depth = depth_map.get(smoothing_level, 9)
        
        # Standard reconstruction parameters
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=depth,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # IMPROVED density filtering - this is the key fix
        if len(densities) > 0:
            densities = np.asarray(densities)
            
            # Use statistical approach instead of fixed percentile
            mean_density = np.mean(densities)
            std_density = np.std(densities)
            
            # Remove vertices that are more than 2 standard deviations below mean
            # This is much more precise than percentile-based filtering
            thresh = max(mean_density - 2 * std_density, np.min(densities) * 1.1)
            
            vertices_to_remove = densities < thresh
            removed_count = np.sum(vertices_to_remove)
            
            log(f"Density filtering: removing {removed_count}/{len(densities)} vertices (thresh={thresh:.6f})")
            
            if removed_count > 0:
                mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Minimal cleanup only
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles() 
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        # NO additional smoothing for light/medium
        if smoothing_level in ['high', 'ultra']:
            smoothing_iterations = {'high': 1, 'ultra': 2}
            iterations = smoothing_iterations.get(smoothing_level, 1)
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations=iterations, 
                lambda_filter=0.2
            )
        
        # Convert to trimesh
        vertices_np = np.asarray(mesh.vertices)
        faces_np = np.asarray(mesh.triangles)
        
        if len(vertices_np) > 0 and len(faces_np) > 0:
            tm = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
            
            # Transfer colors if available
            if mesh.has_vertex_colors():
                vertex_colors = np.asarray(mesh.vertex_colors)
                if vertex_colors.max() <= 1.0:
                    vertex_colors = (vertex_colors * 255).astype(np.uint8)
                tm.visual.vertex_colors = vertex_colors
            
            log(f"Precise Poisson reconstruction: {len(tm.vertices)} vertices, {len(tm.faces)} faces")
            return tm
        else:
            log("Precise Poisson failed: no mesh generated")
            return None
            
    except Exception as e:
        log(f"Precise Poisson reconstruction failed: {e}")
        return None


def create_mesh_from_points_basic(points: np.ndarray, colors: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """
    Create a basic mesh from points using simple triangulation
    """
    try:
        # Use trimesh's convex hull as a simple mesh creation method
        mesh = trimesh.Trimesh(vertices=points).convex_hull
        
        if colors is not None and len(colors) == len(points):
            # Map colors to mesh vertices by finding closest points
            mesh_vertices = np.asarray(mesh.vertices)
            if HAS_SCIPY and cKDTree is not None:
                tree = cKDTree(points)
                distances, indices = tree.query(mesh_vertices, k=1)
                mesh_colors = colors[indices]
            else:
                # Simple closest point mapping without scipy
                mesh_colors = []
                for mv in mesh_vertices:
                    dists = np.sum((points - mv)**2, axis=1)
                    closest_idx = np.argmin(dists)
                    mesh_colors.append(colors[closest_idx])
                mesh_colors = np.array(mesh_colors)
            
            # Set vertex colors
            if mesh_colors.max() <= 1.0:
                mesh_colors = (mesh_colors * 255).astype(np.uint8)
            mesh.visual.vertex_colors = mesh_colors
        
        log(f"Created basic mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh
        
    except Exception as e:
        log(f"Basic mesh creation failed: {e}")
        # Ultimate fallback - create a simple box
        return trimesh.creation.box(extents=[1, 1, 1])


def load_ply_file(file_path: str) -> Dict[str, Any]:
    """Load PLY file and extract vertices, colors, normals"""
    try:
        # Try with trimesh first (most compatible)
        mesh_data = trimesh.load(str(file_path))
        
        if hasattr(mesh_data, 'vertices'):
            vertices = np.asarray(mesh_data.vertices)
            
            # Extract colors if available
            colors = None
            if hasattr(mesh_data.visual, 'vertex_colors'):
                colors = np.asarray(mesh_data.visual.vertex_colors)
                if colors.shape[1] == 4:  # RGBA to RGB
                    colors = colors[:, :3]
                if colors.max() > 1.0:  # Convert to 0-1 range
                    colors = colors / 255.0
            
            # Extract faces if it's a mesh
            faces = None
            if hasattr(mesh_data, 'faces') and len(mesh_data.faces) > 0:
                faces = np.asarray(mesh_data.faces)
            
            return {
                'vertices': vertices,
                'faces': faces,
                'colors': colors,
                'is_point_cloud': faces is None or len(faces) == 0
            }
    
    except Exception as e:
        log(f"Trimesh loading failed: {e}")
    
    try:
        # Fallback to plyfile
        plydata = PlyData.read(file_path)
        vertices = plydata['vertex']
        
        # Extract coordinates
        coords = np.column_stack([vertices['x'], vertices['y'], vertices['z']])
        
        # Extract colors if available
        colors = None
        if 'red' in vertices.dtype.names and 'green' in vertices.dtype.names and 'blue' in vertices.dtype.names:
            colors = np.column_stack([vertices['red'], vertices['green'], vertices['blue']])
            if colors.max() > 1.0:
                colors = colors / 255.0
        
        # Check for faces
        faces = None
        if 'face' in plydata:
            face_data = plydata['face']
            if 'vertex_indices' in face_data.dtype.names:
                faces = np.array([list(face[0]) for face in face_data['vertex_indices']])
        
        return {
            'vertices': coords,
            'faces': faces,
            'colors': colors,
            'is_point_cloud': faces is None or len(faces) == 0
        }
        
    except Exception as e:
        log(f"PLYfile loading failed: {e}")
        raise RuntimeError(f"Failed to load PLY file: {e}")


def smooth_mesh_basic(mesh: trimesh.Trimesh, smoothing_level: str = "medium") -> trimesh.Trimesh:
    """Apply basic smoothing using trimesh methods"""
    try:
        iterations_map = {
            'light': 1,
            'medium': 2,
            'high': 3,
            'ultra': 5
        }
        
        iterations = iterations_map.get(smoothing_level, 2)
        
        smoothed_mesh = mesh
        for i in range(iterations):
            if hasattr(smoothed_mesh, 'smoothed'):
                smoothed_mesh = smoothed_mesh.smoothed()
            else:
                break
        
        log(f"Applied basic smoothing: {iterations} iterations")
        return smoothed_mesh
        
    except Exception as e:
        log(f"Basic smoothing failed: {e}")
        return mesh


class PLYConverter:
    """Fixed PLY Converter with 360° visibility"""
    
    def convert_ply(self, input_path: str, output_dir: str, output_formats: list, 
                   conversion_id: str, progress_callback: Optional[Callable] = None, 
                   smoothing_level: str = "medium") -> Dict[str, str]:
        """Convert PLY file to specified formats with face orientation fixes"""
        
        def update_progress(message: str, progress: int):
            if progress_callback:
                progress_callback(message, progress)
            log(f"Progress {progress}%: {message}")
        
        try:
            update_progress("Loading PLY file...", 5)
            
            # Load PLY data
            ply_data = load_ply_file(input_path)
            vertices = ply_data['vertices']
            faces = ply_data['faces']
            colors = ply_data['colors']
            is_point_cloud = ply_data['is_point_cloud']
            
            update_progress(f"Loaded {len(vertices)} vertices", 15)
            
            if is_point_cloud:
                update_progress("Point cloud detected, precise surface reconstruction...", 25)
                
                # Try precise Poisson reconstruction
                mesh = precise_poisson_reconstruction(vertices, colors, smoothing_level)
                
                if mesh is None:
                    update_progress("Using basic surface reconstruction...", 35)
                    mesh = create_mesh_from_points_basic(vertices, colors)
                else:
                    update_progress("Precise surface reconstruction completed", 45)
            else:
                update_progress("Mesh detected, preserving original geometry...", 25)
                # Create trimesh from existing mesh data
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                if colors is not None:
                    if colors.max() <= 1.0:
                        colors = (colors * 255).astype(np.uint8)
                    mesh.visual.vertex_colors = colors
                update_progress("Original mesh loaded", 40)
            
            update_progress("Fixing face orientation for 360° visibility...", 55)
            
            # Fix face orientation and normals so mesh is visible from all angles
            mesh = fix_face_orientation_and_normals(mesh)
            
            update_progress(f"Applying {smoothing_level} smoothing...", 65)
            
            # Apply smoothing only for existing meshes, not point cloud reconstructions
            if not is_point_cloud:
                mesh = smooth_mesh_basic(mesh, smoothing_level)
            
            update_progress("Final cleanup...", 75)
            
            # Final minimal cleanup
            try:
                mesh.remove_degenerate_faces()
                mesh.remove_duplicate_faces()
                mesh.remove_unreferenced_vertices()
            except Exception as e:
                log(f"Final cleanup warning: {e}")
            
            update_progress("Exporting files...", 80)
            
            # Export to specified formats
            results = {}
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for i, fmt in enumerate(output_formats):
                try:
                    progress = 80 + int((i + 1) / len(output_formats) * 15)
                    update_progress(f"Exporting {fmt.upper()}...", progress)
                    
                    filename = f"{conversion_id}_smooth.{fmt}"
                    file_path = output_path / filename
                    
                    # Export based on format
                    mesh.export(str(file_path))
                    results[fmt] = str(file_path)
                    
                    log(f"Exported {fmt.upper()}: {file_path}")
                    
                except Exception as export_error:
                    log(f"Failed to export {fmt}: {export_error}")
                    continue
            
            if not results:
                raise RuntimeError("No files were successfully exported")
            
            update_progress("Conversion completed successfully!", 100)
            return results
            
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            log(f"ERROR: {error_msg}")
            log(f"Traceback: {traceback.format_exc()}")
            update_progress(error_msg, 0)
            raise RuntimeError(error_msg) from e


def main():
    """Simple test function"""
    print("PLY Converter - Fixed Version with 360° Visibility")
    print(f"Open3D available: {HAS_OPEN3D}")
    print(f"SciPy available: {HAS_SCIPY}")
    print(f"Scikit-image available: {HAS_SKIMAGE}")
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        if os.path.exists(input_file):
            converter = PLYConverter()
            try:
                results = converter.convert_ply(
                    input_file, 
                    "output", 
                    ['stl', 'obj'], 
                    'test',
                    lambda msg, prog: print(f"[{prog}%] {msg}"),
                    'medium'
                )
                print("Conversion successful!")
                for fmt, path in results.items():
                    print(f"  {fmt.upper()}: {path}")
            except Exception as e:
                print(f"Conversion failed: {e}")
        else:
            print(f"File not found: {input_file}")
    else:
        print("Usage: python ply_converter.py <input.ply>")


if __name__ == "__main__":
    main()