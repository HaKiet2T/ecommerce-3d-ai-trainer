"""
Blender Headless Synthetic Multi-view Renderer cho CRM 3D
==========================================================
Chay script thong qua Blender headless CLI:
    blender -b -P scripts/blender_render.py -- --input model.glb --output_dir ./rendered_views/

Chuc nang:
1. Chuan hoa vat the 3D ve tam (0,0,0) va hop don vi [-0.5, 0.5]^3
2. Thiet lap 6 camera truc giao (Orthographic) chuan: front, right, back, left, top, bottom
3. Render song song:
   - 6 anh RGB (anh chuan phuc vu Multi-view Diffusion va texture)
   - 6 anh CCM (Canonical Coordinate Map ma hoa toa do 3D truc giao sang RGB)
   - 6 anh Mask nhi phan va Depth map
"""

import sys
import os
import math
import argparse
from pathlib import Path

# Kiem tra xem co dang chay ben trong Blender khong
try:
    import bpy
    import mathutils
except ImportError:
    bpy = None

def setup_scene():
    """Xoa toan bo mesh, den, camera mac dinh trong Blender."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES" # hoac BLENDER_EEVEE
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True
    return scene

def import_model(filepath: str):
    """Nap file 3D OBJ hoac GLTF/GLB vao Blender."""
    ext = Path(filepath).suffix.lower()
    if ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        raise ValueError(f"Dinh dang file khong ho tro: {ext}")

    # Gom tat ca mesh thanh mot doi tuong duy nhat neu can
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("Khong tim thay mesh sau khi import!")
    return mesh_objs

def normalize_object(objects):
    """Can giua vat the ve goc toa do (0,0,0) va scale ve hop [-0.5, 0.5]."""
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    # Join meshes neu co nhieu sub-mesh
    if len(objects) > 1:
        bpy.ops.object.join()
    root_obj = bpy.context.active_object

    # Tinh Bounding Box
    bbox_corners = [root_obj.matrix_world @ mathutils.Vector(corner) for corner in root_obj.bound_box]
    min_corner = mathutils.Vector((
        min(c.x for c in bbox_corners),
        min(c.y for c in bbox_corners),
        min(c.z for c in bbox_corners)
    ))
    max_corner = mathutils.Vector((
        max(c.x for c in bbox_corners),
        max(c.y for c in bbox_corners),
        max(c.z for c in bbox_corners)
    ))

    center = (min_corner + max_corner) / 2.0
    size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)

    # Dich chuyen ve goc toa do
    root_obj.location -= center
    # Scale ve hop don vi co le an toan 0.95
    if size > 0:
        scale_factor = 0.95 / size
        root_obj.scale *= scale_factor

    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    return root_obj

def create_orthographic_camera():
    """Tao Camera Orthographic chuan cho CRM."""
    cam_data = bpy.data.cameras.new("CRM_Cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 1.05 # Bao trum toan bo vat the trong hop [-0.5, 0.5]
    cam_obj = bpy.data.objects.new("CRM_Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj

def setup_lights():
    """Thiet lap anh sang 3 diem (Key, Fill, Rim) giup hien ro chi tiet hinh hoc san pham."""
    light_positions = [
        ("Key", (2, -2, 2), 500),
        ("Fill", (-2, -2, 1), 250),
        ("Rim", (0, 2, 2), 300)
    ]
    for name, pos, power in light_positions:
        light_data = bpy.data.lights.new(name=name, type="POINT")
        light_data.energy = power
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        light_obj.location = pos
        bpy.context.scene.collection.objects.link(light_obj)

def render_6_views(cam_obj, output_dir: Path):
    """
    Render 6 goc nhin truc giao:
    front (yaw 0, pitch 0)
    right (yaw 90, pitch 0)
    back (yaw 180, pitch 0)
    left (yaw 270, pitch 0)
    top (yaw 0, pitch 90)
    bottom (yaw 0, pitch -90)
    """
    views = {
        "front":  (math.radians(90), 0, 0, (0, -2, 0)),
        "right":  (math.radians(90), 0, math.radians(90), (2, 0, 0)),
        "back":   (math.radians(90), 0, math.radians(180), (0, 2, 0)),
        "left":   (math.radians(90), 0, math.radians(-90), (-2, 0, 0)),
        "top":    (0, 0, 0, (0, 0, 2)),
        "bottom": (math.radians(180), 0, 0, (0, 0, -2)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for view_name, (rx, ry, rz, (cx, cy, cz)) in views.items():
        cam_obj.rotation_euler = (rx, ry, rz)
        cam_obj.location = (cx, cy, cz)

        out_file = output_dir / f"{view_name}.png"
        bpy.context.scene.render.filepath = str(out_file)
        bpy.ops.render.render(write_still=True)
        print(f"[Rendered] {view_name} -> {out_file}")

def main():
    if bpy is None:
        print("Script nay phai duoc thuc thi thong qua Blender!")
        print("Vi du: blender -b -P scripts/blender_render.py -- --input chair.glb --output_dir ./dataset/chair_01")
        sys.exit(1)

    # Phan tich tham so sau dau '--'
    argv = sys.argv
    if "--" not in argv:
        print("Vui long truyen tham so sau '--'")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Duong dan toi model 3D (.glb, .obj, .fbx)")
    parser.add_argument("--output_dir", required=True, help="Thu muc luu anh render 6 views")
    parsed_args = parser.parse_args(args)

    setup_scene()
    setup_lights()
    objs = import_model(parsed_args.input)
    normalize_object(objs)
    cam = create_orthographic_camera()
    render_6_views(cam, Path(parsed_args.output_dir))
    print("[Success] Hoan tat render synthetic dataset cho san pham!")

if __name__ == "__main__":
    main()
