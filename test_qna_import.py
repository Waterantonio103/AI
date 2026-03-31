"""Test qwen_node_assistant.py serialize_node_tree and _build_node_tree_from_json."""

import bpy
import json
import sys
import os

# Remove any registered addon versions from sys.modules so we load from C:\AI
for mod_name in list(sys.modules.keys()):
    if "qwen_node_assistant" in mod_name or "json_mover" in mod_name:
        del sys.modules[mod_name]

addon_path = r"C:\AI"
if addon_path not in sys.path:
    sys.path.insert(0, addon_path)

# Import directly from file path to avoid addon registration issues
import importlib.util

spec_qna = importlib.util.spec_from_file_location(
    "qwen_node_assistant", os.path.join(addon_path, "qwen_node_assistant.py")
)
qna = importlib.util.module_from_spec(spec_qna)
spec_qna.loader.exec_module(qna)

spec_jm = importlib.util.spec_from_file_location(
    "json_mover", os.path.join(addon_path, "json_mover.py")
)
json_mover = importlib.util.module_from_spec(spec_jm)
spec_jm.loader.exec_module(json_mover)

# Clean up
for ng in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

# Load the test JSON
json_path = r"C:\AI\ghibli_grass.json"
with open(json_path, "r") as f:
    original_data = json.load(f)

print("=== Test 1: Build tree with json_mover, serialize with qwen_node_assistant ===")

ng = bpy.data.node_groups.new(name="TestSerialize", type="GeometryNodeTree")
json_mover.build_node_tree(ng, original_data)

serialized = qna.serialize_node_tree(ng)

print(f"tree_type: {serialized.get('tree_type')}")
print(f"tree_name: {serialized.get('tree_name')}")
print(f"nodes: {len(serialized.get('nodes', []))}")
print(f"links: {len(serialized.get('links', []))}")
print(f"interface: {len(serialized.get('interface', []))}")
print(f"embedded_groups: {list(serialized.get('embedded_groups', {}).keys())}")

iface = serialized.get("interface", [])
has_identifiers = all("identifier" in item for item in iface)
print(f"interface items have identifiers: {has_identifiers}")

links = serialized.get("links", [])
has_link_meta = all(
    "from_socket" in lk
    and "to_socket" in lk
    and "from_socket_name" in lk
    and "to_socket_name" in lk
    for lk in links
)
print(f"links have socket identifiers: {has_link_meta}")

nodes = serialized.get("nodes", [])
has_props = any("properties" in n for n in nodes)
has_parents = any("parent" in n for n in nodes)
has_node_items = any("node_items" in n for n in nodes)
print(f"nodes have properties: {has_props}")
print(f"nodes have parent refs: {has_parents}")
print(f"nodes have node_items: {has_node_items}")

embedded = serialized.get("embedded_groups", {})
if "Curve Root" in embedded:
    cr = embedded["Curve Root"]
    print(f"\nCurve Root embedded group:")
    print(f"  tree_type: {cr.get('tree_type')}")
    print(f"  nodes: {len(cr.get('nodes', []))}")
    print(f"  links: {len(cr.get('links', []))}")
    print(f"  interface: {len(cr.get('interface', []))}")
    for item in cr.get("interface", []):
        print(f"    - {item['name']} ({item['identifier']}, {item['socket_type']})")

print(
    "\n=== Test 2: Rebuild tree using qwen_node_assistant._build_node_tree_from_json ==="
)

for ng in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng)

ng2 = bpy.data.node_groups.new(name="TestQnaImport", type="GeometryNodeTree")
qna._build_node_tree_from_json(ng2, serialized)

print(f"Nodes after qna import: {len(ng2.nodes)}")
print(f"Links after qna import: {len(ng2.links)}")

curve_nodes = [
    n
    for n in ng2.nodes
    if n.type == "GROUP" and n.node_tree and n.node_tree.name == "Curve Root"
]
print(f"Curve Root group nodes: {len(curve_nodes)}")

if "Curve Root" in bpy.data.node_groups:
    cr_group = bpy.data.node_groups["Curve Root"]
    iface_out = [s for s in cr_group.interface.items_tree if s.in_out == "OUTPUT"]
    print(f"Curve Root group interface outputs: {len(iface_out)}")
    for s in iface_out:
        print(f"  - {s.name} ({s.identifier}, {s.socket_type})")
else:
    print("ERROR: Curve Root group not found!")

cr_groups = [ng for ng in bpy.data.node_groups if "Curve Root" in ng.name]
print(f"Groups with 'Curve Root' in name: {len(cr_groups)}")
for g in cr_groups:
    print(f"  - '{g.name}'")

print("\n=== Test 3: Position preservation ===")
mismatches = 0
total = 0
for node in ng2.nodes:
    orig = None
    for nd in nodes:
        if nd["name"] == node.name:
            orig = nd
            break
    if not orig:
        continue
    total += 1
    expected_loc = orig.get("location", [0, 0])
    actual_loc = list(node.location)
    diff = (
        (expected_loc[0] - actual_loc[0]) ** 2 + (expected_loc[1] - actual_loc[1]) ** 2
    ) ** 0.5
    if diff > 1.0:
        mismatches += 1
        if mismatches <= 5:
            print(
                f"  MISMATCH: {node.name}: expected={expected_loc}, actual={actual_loc}"
            )

print(f"Position mismatches: {mismatches}/{total}")

print("\n=== Test 4: Link integrity ===")
print(f"Original links: {len(original_data.get('links', []))}")
print(f"Rebuilt links: {len(ng2.links)}")

# Check that key connections exist
link_pairs = set()
for link in ng2.links:
    link_pairs.add((link.from_node.name, link.to_node.name))

# Verify Group Input -> Distribute Points connection exists
gi_to_dist = any("Group Input" in fn and "Distribute" in tn for fn, tn in link_pairs)
print(f"Group Input -> Distribute Points: {'OK' if gi_to_dist else 'MISSING'}")

# Verify Join Geometry -> Group Output connection exists
join_to_go = any(
    "Join Geometry" in fn and "Group Output" in tn for fn, tn in link_pairs
)
print(f"Join Geometry -> Group Output: {'OK' if join_to_go else 'MISSING'}")

print("\n=== ALL TESTS COMPLETE ===")
