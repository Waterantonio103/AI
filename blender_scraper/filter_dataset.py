"""
filter_dataset.py — Filter training_dataset.jsonl for Blender shader/geometry node data.

Usage:
    python filter_dataset.py [input.jsonl] [output.jsonl] [--min-score N] [--verbose]

Defaults:
    input  = output/training_dataset.jsonl
    output = output/blender_nodes_dataset.jsonl
    --min-score 2   (minimum relevance score to keep a record)
"""

import json
import sys
import os
import re
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Blender node type signatures
# These appear as values in node tree JSON exports from Blender add-ons,
# the NodeToPython tool, node preset exporters, etc.
# ---------------------------------------------------------------------------

# Shader node bl_idnames (ShaderNode*)
SHADER_NODE_TYPES = {
    "ShaderNodeBsdfPrincipled", "ShaderNodeBsdfDiffuse", "ShaderNodeBsdfGlossy",
    "ShaderNodeBsdfTransparent", "ShaderNodeBsdfRefraction", "ShaderNodeBsdfGlass",
    "ShaderNodeBsdfTranslucent", "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfVelvet",
    "ShaderNodeBsdfToon", "ShaderNodeBsdfHair", "ShaderNodeBsdfHairPrincipled",
    "ShaderNodeSubsurfaceScattering", "ShaderNodeEmission", "ShaderNodeBackground",
    "ShaderNodeHoldout", "ShaderNodeMixShader", "ShaderNodeAddShader",
    "ShaderNodeVolumePrincipled", "ShaderNodeVolumeAbsorption", "ShaderNodeVolumeScatter",
    "ShaderNodeTexImage", "ShaderNodeTexEnvironment", "ShaderNodeTexSky",
    "ShaderNodeTexNoise", "ShaderNodeTexMusgrave", "ShaderNodeTexVoronoi",
    "ShaderNodeTexWave", "ShaderNodeTexMagic", "ShaderNodeTexChecker",
    "ShaderNodeTexBrick", "ShaderNodeTexGradient", "ShaderNodeTexCoord",
    "ShaderNodeTexPointDensity", "ShaderNodeTexIES",
    "ShaderNodeMapping", "ShaderNodeNormalMap", "ShaderNodeNormal",
    "ShaderNodeBump", "ShaderNodeDisplacement", "ShaderNodeVectorDisplacement",
    "ShaderNodeMath", "ShaderNodeVectorMath", "ShaderNodeMixRGB", "ShaderNodeMix",
    "ShaderNodeRGBCurve", "ShaderNodeVectorCurve", "ShaderNodeFloatCurve",
    "ShaderNodeHueSaturation", "ShaderNodeBrightContrast", "ShaderNodeGamma",
    "ShaderNodeInvert", "ShaderNodeLightPath", "ShaderNodeFresnel",
    "ShaderNodeLayerWeight", "ShaderNodeWireframe", "ShaderNodeObjectInfo",
    "ShaderNodeParticleInfo", "ShaderNodeHairInfo", "ShaderNodePointInfo",
    "ShaderNodeCameraData", "ShaderNodeUVMap", "ShaderNodeAttribute",
    "ShaderNodeValue", "ShaderNodeRGB", "ShaderNodeVertexColor",
    "ShaderNodeSeparateXYZ", "ShaderNodeCombineXYZ", "ShaderNodeSeparateRGB",
    "ShaderNodeCombineRGB", "ShaderNodeSeparateHSV", "ShaderNodeCombineHSV",
    "ShaderNodeSeparateColor", "ShaderNodeCombineColor",
    "ShaderNodeBlackbody", "ShaderNodeWavelength", "ShaderNodeClamp",
    "ShaderNodeMapRange", "ShaderNodeColorRamp", "ShaderNodeGroup",
    "ShaderNodeOutputMaterial", "ShaderNodeOutputWorld", "ShaderNodeOutputLight",
    "ShaderNodeOutputAOV",
}

# Geometry node bl_idnames (GeometryNode*, FunctionNode*)
GEOMETRY_NODE_TYPES = {
    "GeometryNodeGroup", "GeometryNodeViewer",
    # Mesh
    "GeometryNodeMeshLine", "GeometryNodeMeshCircle", "GeometryNodeMeshUVSphere",
    "GeometryNodeMeshIcoSphere", "GeometryNodeMeshCylinder", "GeometryNodeMeshCone",
    "GeometryNodeMeshGrid", "GeometryNodeMeshCube", "GeometryNodeMeshTorus",
    "GeometryNodeExtrudeMesh", "GeometryNodeFlipFaces", "GeometryNodeSplitEdges",
    "GeometryNodeSubdivideMesh", "GeometryNodeSubdivisionSurface",
    "GeometryNodeTriangulate", "GeometryNodeDualMesh", "GeometryNodeFillCurve",
    "GeometryNodeMeshBoolean", "GeometryNodeMeshToCurve", "GeometryNodeMeshToPoints",
    "GeometryNodeMeshToVolume", "GeometryNodeScaleElements", "GeometryNodeDeleteGeometry",
    "GeometryNodeMergeByDistance", "GeometryNodeEdgePathsToCurves",
    "GeometryNodeEdgePathsToSelection",
    # Curve
    "GeometryNodeCurveLength", "GeometryNodeCurveToMesh", "GeometryNodeCurveToPoints",
    "GeometryNodeCurvePrimitiveLine", "GeometryNodeCurvePrimitiveBezierSegment",
    "GeometryNodeCurvePrimitiveCircle", "GeometryNodeCurvePrimitiveQuadrilateral",
    "GeometryNodeCurvePrimitiveQuadraticBezier", "GeometryNodeCurvePrimitiveStar",
    "GeometryNodeCurvePrimitiveSpiral", "GeometryNodeCurveArc",
    "GeometryNodeFillet", "GeometryNodeResampleCurve", "GeometryNodeReverseCurve",
    "GeometryNodeSampleCurve", "GeometryNodeSubdivideCurve", "GeometryNodeTrimCurve",
    "GeometryNodeSetCurveRadius", "GeometryNodeSetCurveTilt", "GeometryNodeSetCurveHandlePositions",
    "GeometryNodeCurveHandleTypeSelection", "GeometryNodeCurveSetHandles",
    # Points
    "GeometryNodePoints", "GeometryNodePointsToVertices", "GeometryNodePointsToVolume",
    "GeometryNodeDistributePointsOnFaces", "GeometryNodeDistributePointsInVolume",
    # Instances
    "GeometryNodeInstanceOnPoints", "GeometryNodeInstancesToPoints",
    "GeometryNodeRealizeInstances", "GeometryNodeRotateInstances",
    "GeometryNodeScaleInstances", "GeometryNodeTranslateInstances",
    "GeometryNodeInputInstanceRotation", "GeometryNodeInputInstanceScale",
    # Attributes
    "GeometryNodeAttributeStatistic", "GeometryNodeAttributeDomainSize",
    "GeometryNodeBlurAttribute", "GeometryNodeCaptureAttribute",
    "GeometryNodeRemoveAttribute", "GeometryNodeStoreNamedAttribute",
    "GeometryNodeNamedAttribute", "GeometryNodeInputNamedLayerSelection",
    # Utilities
    "GeometryNodeJoinGeometry", "GeometryNodeGeometryToInstance",
    "GeometryNodeSeparateGeometry", "GeometryNodeConvexHull",
    "GeometryNodeBoundBox", "GeometryNodeTransform", "GeometryNodeSeparateComponents",
    "GeometryNodeProximity", "GeometryNodeRaycast", "GeometryNodeSampleIndex",
    "GeometryNodeSampleNearest",
    # Volume
    "GeometryNodeVolumeCube", "GeometryNodeVolumeToMesh",
    # Material
    "GeometryNodeReplaceMaterial", "GeometryNodeSetMaterial",
    "GeometryNodeSetMaterialIndex", "GeometryNodeMaterialSelection",
    "GeometryNodeInputMaterial", "GeometryNodeInputMaterialIndex",
    # Input
    "GeometryNodeCollectionInfo", "GeometryNodeObjectInfo",
    "GeometryNodeIsViewport", "GeometryNodeSelfObject",
    "GeometryNodeInputID", "GeometryNodeInputIndex",
    "GeometryNodeInputNormal", "GeometryNodeInputPosition",
    "GeometryNodeInputRadius", "GeometryNodeInputEdgeAngle",
    "GeometryNodeInputEdgeVertices", "GeometryNodeInputMeshEdgeNeighbors",
    "GeometryNodeInputMeshFaceArea", "GeometryNodeInputMeshFaceNeighbors",
    "GeometryNodeInputMeshIsland", "GeometryNodeInputMeshVertexNeighbors",
    "GeometryNodeInputShadeSmooth", "GeometryNodeInputCurveHandlePositions",
    "GeometryNodeInputTangent", "GeometryNodeInputCurveTilt",
    "GeometryNodeInputCurveParameter", "GeometryNodeSplineLength",
    "GeometryNodeSplineParameter", "GeometryNodeInputSplineCyclic",
    "GeometryNodeInputSplineResolution",
    # Function nodes (used in GN)
    "FunctionNodeInputBool", "FunctionNodeInputColor", "FunctionNodeInputInt",
    "FunctionNodeInputString", "FunctionNodeInputVector", "FunctionNodeInputSpecialCharacters",
    "FunctionNodeRandomValue", "FunctionNodeAlignEulerToVector",
    "FunctionNodeBooleanMath", "FunctionNodeCompare", "FunctionNodeFloatToInt",
    "FunctionNodeInputRotation", "FunctionNodeRotateEuler",
    "FunctionNodeReplaceString", "FunctionNodeSliceString",
    "FunctionNodeStringLength", "FunctionNodeStringToCurves",
    "FunctionNodeValueToString", "FunctionNodeFindInString",
}

ALL_NODE_TYPES = SHADER_NODE_TYPES | GEOMETRY_NODE_TYPES

# Node tree type identifiers (appear as string values in exported JSONs)
NODE_TREE_TYPES = {
    "ShaderNodeTree", "GeometryNodeTree", "CompositorNodeTree",
    "TextureNodeTree", "shader", "geometry",
}

# Path keywords that strongly suggest Blender node content
PATH_KEYWORDS = [
    "blender", "shader", "geometry_node", "geometry-node", "nodetree",
    "node_tree", "node-tree", "material", "bsdf", "principled",
    "node_group", "nodegroup", "geonodes", "geo_node",
]

# Content keywords found in Blender node JSON exports
CONTENT_KEYWORDS = [
    "node_tree", "nodetree", "ShaderNode", "GeometryNode", "FunctionNode",
    "bl_idname", "node_group", "bsdf", "principled", "blender",
    "use_nodes", "cycles", "eevee", "material_output",
]

# Keywords that strongly indicate NON-Blender content (Unity, Unreal, etc.)
DISCARD_KEYWORDS = [
    "unity", "unityengine", "monobehaviour", "UnityEditor",
    "serializedversion", "m_component", "m_GameObject", "m_Script",
    "fileFormatVersion", "guid", "AssetImporter", "PixelFormat",
    "unreal", "uasset", "blueprint", "UObject",
    "minecraft", "forge", "fabricmc",
    "godot", "gdscript",
    "three.js", "threejs", "WebGLRenderer",
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_record(record: dict) -> tuple[int, list[str]]:
    """
    Return (score, reasons). score >= min_score means keep.
    Higher score = more confident it's Blender shader/geometry node data.
    """
    score = 0
    reasons = []
    path    = (record.get("path") or "").lower()
    repo    = (record.get("repo") or "").lower()
    content = record.get("content")

    # --- Hard discard: known non-Blender signatures in content ---
    content_str = json.dumps(content) if not isinstance(content, str) else content
    content_lower = content_str.lower()

    for kw in DISCARD_KEYWORDS:
        if kw.lower() in content_lower[:4000]:  # check first 4KB only
            return -1, [f"discard:{kw}"]

    # --- Path-based scoring ---
    for kw in PATH_KEYWORDS:
        if kw in path or kw in repo:
            score += 2
            reasons.append(f"path:{kw}")
            break  # one path match is enough

    if path.endswith(".json"):
        # Extra weight for paths that look like node preset files
        if any(x in path for x in ("node", "shader", "material", "geo")):
            score += 1
            reasons.append("path:node-related-name")

    # --- Content-based scoring ---
    # Check for exact node type names (highest confidence)
    content_sample = content_str[:50000]  # cap to avoid huge files
    for node_type in ALL_NODE_TYPES:
        if node_type in content_sample:
            score += 5
            reasons.append(f"node_type:{node_type}")
            break  # one hit is enough for max points

    # Check for tree type identifiers
    for tree_type in NODE_TREE_TYPES:
        if tree_type in content_sample:
            score += 3
            reasons.append(f"tree_type:{tree_type}")
            break

    # Check for generic Blender content keywords
    kw_hits = 0
    for kw in CONTENT_KEYWORDS:
        if kw.lower() in content_lower[:20000]:
            kw_hits += 1
    if kw_hits >= 3:
        score += 3
        reasons.append(f"content_keywords:{kw_hits}")
    elif kw_hits >= 1:
        score += 1
        reasons.append(f"content_keywords:{kw_hits}")

    # Structural check: does the JSON look like a node tree?
    if isinstance(content, dict):
        keys = set(content.keys())
        node_struct_keys = {"nodes", "links", "inputs", "outputs", "node_tree",
                            "nodetree", "node_group", "type", "bl_idname"}
        overlap = keys & node_struct_keys
        if len(overlap) >= 3:
            score += 3
            reasons.append(f"struct:{overlap}")
        elif len(overlap) >= 1:
            score += 1
            reasons.append(f"struct:{overlap}")

        # Top-level "nodes" array whose items have a "type" key
        if "nodes" in content and isinstance(content["nodes"], list):
            sample_nodes = content["nodes"][:5]
            if any(isinstance(n, dict) and "type" in n for n in sample_nodes):
                score += 2
                reasons.append("nodes_array_with_type")

    return score, reasons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter JSONL for Blender node data")
    parser.add_argument("input",  nargs="?", default="output/training_dataset.jsonl")
    parser.add_argument("output", nargs="?", default="output/blender_nodes_dataset.jsonl")
    parser.add_argument("--min-score", type=int, default=2,
                        help="Minimum relevance score to keep a record (default: 2)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print score and reasons for each kept record")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't write output, just print stats")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[error] Input file not found: {args.input}")
        sys.exit(1)

    input_size_mb = os.path.getsize(args.input) / 1024 / 1024
    print(f"Input:      {args.input}  ({input_size_mb:.1f} MB)")
    print(f"Output:     {args.output}")
    print(f"Min score:  {args.min_score}")
    print()

    # --- Load already-kept records so we never write duplicates ---
    existing_keys: set[str] = set()
    if os.path.exists(args.output) and not args.dry_run:
        print(f"Loading existing records from {args.output} ...", end=" ", flush=True)
        with open(args.output, "r", encoding="utf-8") as ef:
            for line in ef:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec.get("repo", "") + "|" + rec.get("path", ""))
                    existing_keys.add(key)
                except json.JSONDecodeError:
                    pass
        print(f"{len(existing_keys):,} existing records found — will skip duplicates.\n")

    total     = 0
    kept      = 0
    skipped_dup = 0
    discarded = 0
    errors    = 0
    score_dist = {}  # score -> count

    # Append to existing file instead of overwriting
    out_f = open(args.output, "a", encoding="utf-8") if not args.dry_run else None

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                # Skip if already in the dataset
                key = (record.get("repo", "") + "|" + record.get("path", ""))
                if key in existing_keys:
                    skipped_dup += 1
                    continue

                score, reasons = score_record(record)
                score_dist[score] = score_dist.get(score, 0) + 1

                if score >= args.min_score:
                    kept += 1
                    existing_keys.add(key)  # prevent dupes within this run too
                    if out_f:
                        record["_score"]   = score
                        record["_reasons"] = reasons
                        out_f.write(json.dumps(record) + "\n")
                    if args.verbose:
                        print(f"  KEEP  score={score:2d}  {record.get('repo','')} / {record.get('path','')}")
                        print(f"        reasons: {reasons}")
                else:
                    discarded += 1

                if total % 10000 == 0:
                    pct = kept / total * 100 if total else 0
                    print(f"  {total:>7,} scanned  |  {kept:>6,} new  |  "
                          f"{skipped_dup:>6,} dupes skipped  |  {discarded:>6,} discarded")

    finally:
        if out_f:
            out_f.close()

    # Final stats
    output_mb = os.path.getsize(args.output) / 1024 / 1024 if (not args.dry_run and os.path.exists(args.output)) else 0
    new_pct = kept / max(1, total - skipped_dup) * 100
    print(f"""
=== Filter Complete ===
  Total records:     {total:,}
  Already in dataset:{skipped_dup:,}  (skipped)
  New records added: {kept:,}  ({new_pct:.1f}% of new input)
  Discarded:         {discarded:,}
  Parse errors:      {errors}
  Dataset size now:  {output_mb:.1f} MB
  Output file:     {args.output}

Score distribution:
""")
    for s in sorted(score_dist.keys()):
        bar = "#" * min(50, score_dist[s] // max(1, total // 500))
        print(f"  score {s:3d}: {score_dist[s]:>7,}  {bar}")

    print(f"""
Tips:
  Too few results?  Lower --min-score to 1
  Too much noise?   Raise --min-score to 4 or 5
  See what's kept?  Add --verbose
""")

if __name__ == "__main__":
    main()
