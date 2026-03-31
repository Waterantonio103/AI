import bpy
import json
import sys
import os

# Add the directory containing json_mover.py to the path
addon_path = r"C:\AI"
if addon_path not in sys.path:
    sys.path.append(addon_path)

# Import the addon module
import json_mover

# Clean up any existing data
for ng in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

# Read the test JSON
json_path = r"C:\AI\ghibli_grass.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Create a new GeometryNodeTree and import
ng = bpy.data.node_groups.new(name="TestImport", type="GeometryNodeTree")
json_mover.build_node_tree(ng, data)

# Verify the Curve Root nodes
print("=== Verification ===")
print(f"Total nodes in tree: {len(ng.nodes)}")

curve_root_nodes = [
    n
    for n in ng.nodes
    if n.type == "GROUP" and n.node_tree and n.node_tree.name == "Curve Root"
]
print(f"Curve Root nodes found: {len(curve_root_nodes)}")

for i, node in enumerate(curve_root_nodes):
    print(f"\nCurve Root node {i}: '{node.name}'")
    print(f"  - use_custom_color: {node.use_custom_color}")
    print(f"  - color: {list(node.color)}")
    print(f"  - node_tree: {node.node_tree.name if node.node_tree else 'None'}")
    print(f"  - outputs count: {len(node.outputs)}")
    for sock in node.outputs:
        print(f"    Output: '{sock.name}' (identifier: {sock.identifier})")

# Verify the embedded group was created correctly
if "Curve Root" in bpy.data.node_groups:
    curve_root_group = bpy.data.node_groups["Curve Root"]
    print(f"\nEmbedded 'Curve Root' group:")
    print(f"  - Type: {curve_root_group.type}")
    print(
        f"  - Interface outputs: {len([s for s in curve_root_group.interface.items_tree if s.in_out == 'OUTPUT'])}"
    )
    for item in curve_root_group.interface.items_tree:
        if item.in_out == "OUTPUT":
            print(
                f"    - {item.name} (identifier: {item.identifier}, type: {item.socket_type})"
            )
else:
    print("\nERROR: 'Curve Root' embedded group not found!")

# Check for duplicate node group issue
curve_root_groups = [ng for ng in bpy.data.node_groups if "Curve Root" in ng.name]
print(f"\nNode groups with 'Curve Root' in name: {len(curve_root_groups)}")
for grp in curve_root_groups:
    print(f"  - '{grp.name}'")

print("\n=== Test Complete ===")
