"""
Test script to verify the Curve Root node import fix in json_mover.py

Run this in Blender's Scripting workspace (Text Editor > Run Script).
It will:
1. Import the json_mover addon
2. Load the ghibli_grass.json file
3. Import the node tree
4. Verify the Curve Root node and its connections are properly restored
5. Report results in the system console
"""

import bpy
import json
import os
import sys

# Path to the JSON file and addon
SCRIPT_DIR = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else "C:/AI"
JSON_FILE = "C:/AI/ghibli_grass.json"
ADDON_FILE = "C:/AI/json_mover.py"


def test_curve_root_import():
    """Test that Curve Root nodes import correctly with proper socket connections."""

    results = {"passed": 0, "failed": 0, "errors": []}

    def check(condition, message):
        if condition:
            results["passed"] += 1
            print(f"  PASS: {message}")
        else:
            results["failed"] += 1
            results["errors"].append(message)
            print(f"  FAIL: {message}")

    # Clean up any existing data
    for ng in list(bpy.data.node_groups):
        if ng.name == "ghibli_grass" or ng.name == "Curve Root":
            bpy.data.node_groups.remove(ng)

    # Load and execute the addon
    print("\n" + "=" * 60)
    print("TEST: Curve Root Node Import Fix")
    print("=" * 60)

    print("\n1. Loading json_mover addon...")
    try:
        # Read and execute the addon code
        with open(ADDON_FILE, "r") as f:
            addon_code = f.read()
        exec(addon_code)
        print("  Addon loaded successfully")
    except Exception as e:
        print(f"  ERROR loading addon: {e}")
        return results

    # Check JSON file exists
    print(f"\n2. Checking JSON file: {JSON_FILE}")
    check(os.path.exists(JSON_FILE), f"JSON file exists at {JSON_FILE}")
    if not os.path.exists(JSON_FILE):
        return results

    # Load JSON data
    print("\n3. Loading JSON data...")
    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
        check("tree_type" in data, "JSON has tree_type")
        check(data["tree_type"] == "GEOMETRY", "tree_type is GEOMETRY")
        check("embedded_groups" in data, "JSON has embedded_groups")
        check(
            "Curve Root" in data.get("embedded_groups", {}),
            "Embedded 'Curve Root' group exists",
        )
    except Exception as e:
        print(f"  ERROR loading JSON: {e}")
        return results

    # Import the node tree using the addon's build_node_tree function
    print("\n4. Creating node tree and importing...")
    try:
        ng = bpy.data.node_groups.new(name="ghibli_grass", type="GeometryNodeTree")

        # Call build_node_tree directly
        from json_mover import build_node_tree

        interface_id_map = build_node_tree(ng, data)

        check(len(ng.nodes) > 0, "Node tree has nodes after import")
        print(f"  Created {len(ng.nodes)} nodes")
    except Exception as e:
        print(f"  ERROR building node tree: {e}")
        import traceback

        traceback.print_exc()
        return results

    # Verify Curve Root node exists and is properly configured
    print("\n5. Verifying Curve Root node...")
    curve_root_node = None
    curve_root_001_node = None

    for node in ng.nodes:
        if node.name == "Curve Root" and node.bl_idname == "GeometryNodeGroup":
            curve_root_node = node
        elif node.name == "Curve Root.001" and node.bl_idname == "GeometryNodeGroup":
            curve_root_001_node = node

    check(curve_root_node is not None, "Curve Root node exists")
    check(curve_root_001_node is not None, "Curve Root.001 node exists")

    if curve_root_node:
        check(
            curve_root_node.node_tree is not None,
            "Curve Root node has node_tree assigned",
        )
        if curve_root_node.node_tree:
            check(
                curve_root_node.node_tree.name == "Curve Root",
                f"Curve Root node_tree name is 'Curve Root' (got: {curve_root_node.node_tree.name})",
            )

            # Check that the node has outputs (should have 4 outputs from the embedded group)
            check(
                len(curve_root_node.outputs) > 0,
                f"Curve Root node has outputs (count: {len(curve_root_node.outputs)})",
            )

            # Print output socket identifiers for debugging
            print("  Output sockets:")
            for sock in curve_root_node.outputs:
                print(f"    - {sock.name} (identifier: {sock.identifier})")

    if curve_root_001_node:
        check(
            curve_root_001_node.node_tree is not None,
            "Curve Root.001 node has node_tree assigned",
        )
        if curve_root_001_node.node_tree:
            check(
                curve_root_001_node.node_tree.name == "Curve Root",
                f"Curve Root.001 node_tree name is 'Curve Root' (got: {curve_root_001_node.node_tree.name})",
            )

    # Verify the embedded Curve Root group exists and has correct structure
    print("\n6. Verifying embedded Curve Root group...")
    curve_root_group = bpy.data.node_groups.get("Curve Root")
    check(
        curve_root_group is not None,
        "Curve Root node group exists in bpy.data.node_groups",
    )

    if curve_root_group:
        check(
            curve_root_group.type == "GEOMETRY",
            f"Curve Root group type is GEOMETRY (got: {curve_root_group.type})",
        )
        check(
            len(curve_root_group.nodes) > 0,
            f"Curve Root group has nodes (count: {len(curve_root_group.nodes)})",
        )
        check(
            len(curve_root_group.interface.items_tree) > 0,
            f"Curve Root group has interface sockets (count: {len(curve_root_group.interface.items_tree)})",
        )

        # Print interface sockets for debugging
        print("  Interface sockets:")
        for item in curve_root_group.interface.items_tree:
            if hasattr(item, "socket_type"):
                print(
                    f"    - {item.name} (identifier: {item.identifier}, in_out: {item.in_out})"
                )

    # Verify links are created correctly
    print("\n7. Verifying node links...")
    link_count = len(ng.links)
    check(link_count > 0, f"Node tree has links (count: {link_count})")

    # Specifically check links involving Curve Root nodes
    curve_root_links = [
        l
        for l in ng.links
        if l.from_node.name in ("Curve Root", "Curve Root.001")
        or l.to_node.name in ("Curve Root", "Curve Root.001")
    ]
    check(
        len(curve_root_links) > 0,
        f"Curve Root nodes have connections (count: {len(curve_root_links)})",
    )

    if curve_root_links:
        print("  Curve Root links:")
        for link in curve_root_links:
            print(
                f"    {link.from_node.name}:{link.from_socket.name} -> {link.to_node.name}:{link.to_socket.name}"
            )

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
    if results["errors"]:
        print("\nFailed checks:")
        for err in results["errors"]:
            print(f"  - {err}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    test_curve_root_import()
