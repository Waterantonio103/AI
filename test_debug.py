import bpy
import sys

sys.path.append(r"C:\AI")
import json_mover
import json

for ng in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)

with open(r"C:\AI\ghibli_grass.json") as f:
    data = json.load(f)

ng = bpy.data.node_groups.new(name="Test", type="GeometryNodeTree")
json_mover.build_node_tree(ng, data)

# Check CaptureAttribute attributes
for node in ng.nodes:
    if "CaptureAttribute" in node.bl_idname:
        print(f"Node: {node.name} ({node.bl_idname})")
        # List all non-private attributes
        attrs = [
            a
            for a in dir(node)
            if not a.startswith("_") and not callable(getattr(node, a, None))
        ]
        for a in sorted(attrs):
            try:
                v = getattr(node, a)
                if not isinstance(v, (bpy.types.bpy_struct, bpy.types.PropertyGroup)):
                    print(f"  {a}: {v}")
            except:
                pass
        break

# Check StoreNamedAttribute attributes
for node in ng.nodes:
    if "StoreNamedAttribute" in node.bl_idname:
        print(f"\nNode: {node.name} ({node.bl_idname})")
        attrs = [
            a
            for a in dir(node)
            if not a.startswith("_") and not callable(getattr(node, a, None))
        ]
        for a in sorted(attrs):
            try:
                v = getattr(node, a)
                if not isinstance(v, (bpy.types.bpy_struct, bpy.types.PropertyGroup)):
                    print(f"  {a}: {v}")
            except:
                pass
        break

# Check what node_items collections exist
print("\n=== node_items collections ===")
items_collections = ["capture_items", "store_items", "repeat_items", "simulation_items"]
for node in ng.nodes:
    for col_name in items_collections:
        if hasattr(node, col_name):
            col = getattr(node, col_name)
            print(f"{node.name} ({node.bl_idname}).{col_name}: {len(col)} items")
