import bpy
import json
import sys
import math

addon_path = r"C:\AI"
if addon_path not in sys.path:
    sys.path.append(addon_path)

import importlib
import json_mover

importlib.reload(json_mover)

# Clean up
for ng in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng)
for mat in list(bpy.data.materials):
    bpy.data.materials.remove(mat)
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj)

with open(r"C:\AI\ghibli_grass.json", "r") as f:
    original_data = json.load(f)

# Build a map of expected absolute positions from the JSON
# location_absolute is the absolute position stored in properties
expected_positions = {}
for nd in original_data.get("nodes", []):
    loc_abs = nd.get("properties", {}).get("location_absolute")
    if loc_abs:
        expected_positions[nd["name"]] = loc_abs

print(f"Expected absolute positions for {len(expected_positions)} nodes")

# Import
ng = bpy.data.node_groups.new(name="TestImport", type="GeometryNodeTree")
json_mover.build_node_tree(ng, original_data)

# Check actual absolute positions
print("\n=== Position verification (first 10 nodes) ===")
mismatches = 0
checked = 0
for node in ng.nodes:
    if node.name not in expected_positions:
        continue
    checked += 1
    if checked > 10:
        break
    expected = expected_positions[node.name]
    actual = list(node.location_absolute)
    diff = math.sqrt((expected[0] - actual[0]) ** 2 + (expected[1] - actual[1]) ** 2)
    status = "OK" if diff < 1.0 else f"OFF by {diff:.1f}"
    if diff >= 1.0:
        mismatches += 1
    print(f"  {node.name}: expected={expected}, actual={actual} [{status}]")

# Full count
mismatches = 0
total = 0
for node in ng.nodes:
    if node.name not in expected_positions:
        continue
    total += 1
    expected = expected_positions[node.name]
    actual = list(node.location_absolute)
    diff = math.sqrt((expected[0] - actual[0]) ** 2 + (expected[1] - actual[1]) ** 2)
    if diff >= 1.0:
        mismatches += 1

print(f"\nTotal: {mismatches}/{total} nodes mismatched (>1 pixel)")

# Re-export and re-import
reexported = json_mover.extract_node_tree(ng, "GEOMETRY")

for ng2 in list(bpy.data.node_groups):
    bpy.data.node_groups.remove(ng2)

ng2 = bpy.data.node_groups.new(name="TestImport2", type="GeometryNodeTree")
json_mover.build_node_tree(ng2, reexported)

print("\n=== After round-trip (first 10 nodes) ===")
mismatches2 = 0
checked2 = 0
for node in ng2.nodes:
    if node.name not in expected_positions:
        continue
    checked2 += 1
    if checked2 > 10:
        break
    expected = expected_positions[node.name]
    actual = list(node.location_absolute)
    diff = math.sqrt((expected[0] - actual[0]) ** 2 + (expected[1] - actual[1]) ** 2)
    status = "OK" if diff < 1.0 else f"OFF by {diff:.1f}"
    if diff >= 1.0:
        mismatches2 += 1
    print(f"  {node.name}: expected={expected}, actual={actual} [{status}]")

mismatches2 = 0
total2 = 0
for node in ng2.nodes:
    if node.name not in expected_positions:
        continue
    total2 += 1
    expected = expected_positions[node.name]
    actual = list(node.location_absolute)
    diff = math.sqrt((expected[0] - actual[0]) ** 2 + (expected[1] - actual[1]) ** 2)
    if diff >= 1.0:
        mismatches2 += 1

print(f"\nAfter round-trip: {mismatches2}/{total2} nodes mismatched (>1 pixel)")
print("\n=== DONE ===")
