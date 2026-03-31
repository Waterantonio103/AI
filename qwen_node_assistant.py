# ============================================================================
#  Qwen Node Assistant v3.0 — Blender 5.0+ Addon
#  Connects to a local Ollama instance to analyze AND generate
#  Shader and Geometry Node trees.
#
#  Generation uses a 2-step constrained pipeline:
#    Step 1: LLM picks nodes (schema-enforced JSON)
#    Step 2: Addon builds nodes, reads REAL sockets from Blender,
#            then LLM connects them using only valid socket names
# ============================================================================

bl_info = {
    "name": "Qwen Node Assistant",
    "author": "Claude / Anthropic",
    "version": (3, 0, 0),
    "blender": (5, 0, 0),
    "location": "Node Editor > Sidebar > Qwen Assistant",
    "description": "Analyze and generate shader/geometry node trees with a local Ollama LLM",
    "category": "Node",
}

# === IMPORTS ===

import bpy
import json
import re
import threading
import textwrap
import time
import traceback
import urllib.request
import urllib.error
from collections import OrderedDict


# === GLOBAL STATE ===


class _State:
    """Mutable singleton for cross-thread communication."""

    # Display
    response_lines: list[str] = []
    raw_response: str = ""
    is_running: bool = False
    error: str = ""
    dirty: bool = False
    build_log: list[str] = []

    # --- Constrained generation phases ---
    # Phase flow: idle → nodes → building → linking → done
    gen_phase: str = "idle"
    pending_phase1: dict | None = None  # nodes spec from step 1
    pending_phase2: dict | None = None  # links spec from step 2
    # Context passed between phases
    gen_user_prompt: str = ""
    gen_base_url: str = ""
    gen_model: str = ""
    gen_tree_type: str = ""  # "ShaderNodeTree" or "GeometryNodeTree"
    gen_node_ids: list[str] = []  # IDs of nodes built in phase 1
    gen_socket_info: str = ""  # formatted socket info for phase 2 prompt
    gen_clear: bool = True  # whether to clear tree before building
    abort_requested: bool = False  # set True by Abort operator to stop bg thread

    # --- Revert / undo snapshot ---
    pre_gen_snapshot: dict | None = None  # serialized tree state before last generation

    # --- Panel visibility / VRAM unload tracking ---
    panel_last_drawn: float = 0.0  # time.time() stamp updated every panel draw
    model_is_loaded: bool = False  # True after any Ollama request is dispatched
    loaded_model: str = ""  # last model sent to Ollama
    loaded_base_url: str = ""  # base URL used for that model


_state = _State()


# === ADDON PREFERENCES ===


class QNA_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__ or __name__

    ollama_base_url: bpy.props.StringProperty(
        name="Ollama Base URL",
        description="Base URL for the local Ollama API (no trailing slash)",
        default="http://localhost:11434",
    )  # type: ignore

    ollama_model_analyze: bpy.props.StringProperty(
        name="Analyze Model",
        description="Model for analyzing / answering questions about node trees",
        default="qwen2.5vl:7b",
    )  # type: ignore

    ollama_model_generate: bpy.props.StringProperty(
        name="Generate Model",
        description="Model for generating node setups (larger = better)",
        default="qwen2.5:14b",
    )  # type: ignore

    response_wrap_width: bpy.props.IntProperty(
        name="Wrap Width (chars)",
        description="Character width for response text wrapping",
        default=62,
        min=30,
        max=160,
    )  # type: ignore

    gen_clear_existing: bpy.props.BoolProperty(
        name="Clear Tree Before Generating",
        description="Remove all existing nodes before placing generated ones",
        default=True,
    )  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "ollama_base_url")
        layout.separator()
        layout.label(text="Models:")
        layout.prop(self, "ollama_model_analyze")
        layout.prop(self, "ollama_model_generate")
        layout.separator()
        layout.prop(self, "response_wrap_width")
        layout.prop(self, "gen_clear_existing")


def _get_prefs():
    addon_id = __package__ or __name__
    entry = bpy.context.preferences.addons.get(addon_id, None)
    return entry.preferences if entry else None


def _pref(attr: str, fallback):
    p = _get_prefs()
    return getattr(p, attr, fallback) if p else fallback


# === MODEL KNOWLEDGE BASE & CACHE ===

# Profile list — ordered by overall quality (used to rank auto-selection).
# gen/ana: lower = more preferred for that role; 0 = do not recommend.
_MODEL_PROFILES = [
    {
        "pat": "blender-nodes",
        "label": "Blender Nodes (custom)",
        "desc": "Your locally fine-tuned Qwen2.5-3B-Instruct — trained specifically on Blender node graphs.",
        "pros": "Fine-tuned on Blender node data · tiny VRAM (~2 GB) · fast",
        "cons": "3B base — may need more correction passes than larger models",
        "gen": 2,
        "ana": 4,
    },
    {
        "pat": "qwen3",
        "label": "Qwen3",
        "desc": "Alibaba's latest generation. Best reasoning + JSON compliance in the Qwen family.",
        "pros": "Strong reasoning · reliable JSON · good material/shading knowledge",
        "cons": "Slightly slower than Qwen 2.5 at the same parameter count",
        "gen": 1,
        "ana": 1,
    },
    {
        "pat": "mistral-small3.1",
        "label": "Mistral Small 3.1",
        "desc": "24B — most capable model that fits in 16 GB VRAM. Rich material/shading domain knowledge.",
        "pros": "Best understanding of vague descriptions · strong creative output",
        "cons": "~15 GB VRAM — leaves little headroom; noticeably slower",
        "gen": 2,
        "ana": 2,
    },
    {
        "pat": "phi4",
        "label": "Phi-4",
        "desc": "Microsoft Research 14B. Exceptional schema adherence and very low hallucination rate.",
        "pros": "Precise structured output · low hallucination · fast",
        "cons": "Weaker on open-ended material descriptions and shading concepts",
        "gen": 3,
        "ana": 5,
    },
    {
        "pat": "qwen2.5-coder",
        "label": "Qwen2.5-Coder",
        "desc": "Code-tuned 14B. Most reliable JSON output but limited shading domain knowledge.",
        "pros": "Consistent valid JSON · fast · low memory (~9 GB)",
        "cons": "Trained on code not materials — misses nuanced descriptions",
        "gen": 4,
        "ana": 6,
    },
    {
        "pat": "gemma3",
        "label": "Gemma 3",
        "desc": "Google 12B. Outstanding at explaining and reasoning about structured data.",
        "pros": "Excellent analysis · strong instruction-following · good explanations",
        "cons": "Average JSON output quality — better for analysis than generation",
        "gen": 5,
        "ana": 4,
    },
    {
        "pat": "deepseek-r1",
        "label": "DeepSeek-R1",
        "desc": "Shows chain-of-thought reasoning. Best for understanding why a node setup works.",
        "pros": "Transparent reasoning · great for debugging node graphs",
        "cons": "3–5× slower due to thinking tokens · overkill for generation",
        "gen": 6,
        "ana": 3,
    },
    {
        "pat": "qwen2.5",
        "label": "Qwen2.5",
        "desc": "General-purpose Qwen 2.5. Better domain knowledge than coder variant for analysis.",
        "pros": "Good speed/quality balance · wide knowledge base",
        "cons": "Outclassed by Qwen3 at same size · less consistent JSON than coder",
        "gen": 7,
        "ana": 7,
    },
    {
        "pat": "llama3",
        "label": "LLaMA 3",
        "desc": "Meta's general model. Decent analysis but unreliable structured JSON output.",
        "pros": "Good general knowledge · widely supported",
        "cons": "Inconsistent JSON formatting — not ideal for node generation",
        "gen": 8,
        "ana": 9,
    },
    {
        "pat": "qwen2.5vl",
        "label": "Qwen2.5-VL",
        "desc": "Vision-language model. Vision capability is unused in this addon (text-only).",
        "pros": "Fast · small footprint (~5 GB)",
        "cons": "Vision wasted here · 7B reasoning depth limits analysis quality",
        "gen": 9,
        "ana": 8,
    },
    {
        "pat": "mistral",
        "label": "Mistral",
        "desc": "Older 7B base. Fast but limited for complex node setups.",
        "pros": "Very fast · low VRAM",
        "cons": "Limited instruction-following for structured/schema tasks",
        "gen": 10,
        "ana": 10,
    },
]

# Keywords that identify non-generative (embedding) models — filter these out.
_EMBED_KEYWORDS = ("embed", "nomic", "all-minilm", "bge-", "e5-", "rerank")

# Two role-specific item lists kept alive at module level (Blender GC requirement).
# Generate list: sorted by gen rank, excludes models unsuitable for generation.
# Analyze list:  sorted by ana rank, excludes models unsuitable for analysis.
_model_items_generate: list[tuple[str, str, str]] = [
    (
        "qwen2.5-coder:14b",
        "qwen2.5-coder:14b",
        "Code-tuned 14B · consistent JSON output · fast (~9 GB VRAM)",
    ),
]
_model_items_analyze: list[tuple[str, str, str]] = [
    (
        "qwen2.5vl:7b",
        "qwen2.5vl:7b",
        "Vision-language 7B · fast but limited reasoning depth",
    ),
]

# Models with rank above this threshold for a role are excluded from that role's list.
# They will still appear in the other role's list if suitable.
_ROLE_RANK_THRESHOLD = 8


def _is_embed_model(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in _EMBED_KEYWORDS)


def _get_model_profile(name: str) -> dict | None:
    """Return the best-matching profile entry for a model name, or None."""
    n = name.lower()
    for p in _MODEL_PROFILES:
        if p["pat"] in n:
            return p
    return None


def _build_model_items_for_role(
    names: list[str], role: str
) -> list[tuple[str, str, str]]:
    """
    Build a role-specific EnumProperty items list from raw Ollama model names.
    - Filters embedding models.
    - Excludes models ranked above _ROLE_RANK_THRESHOLD for this role
      (unless they are the only option — never leave the list empty).
    - Sorts best-for-role first.
    role: "gen" or "ana"
    """
    key = "gen" if role == "gen" else "ana"
    _FALLBACK = [("none", "(no models — click Refresh)", "")]

    scored = []
    for n in names:
        if _is_embed_model(n):
            continue
        profile = _get_model_profile(n)
        rank = profile[key] if profile else 99
        tooltip = profile["desc"] if profile else f"{n} · Unknown model"
        scored.append((rank, n, tooltip))

    scored.sort()

    # Try to include only models within the threshold
    suitable = [(n, n, tip) for rank, n, tip in scored if rank <= _ROLE_RANK_THRESHOLD]
    if suitable:
        return suitable

    # Fall back to full sorted list if nothing met the threshold
    full = [(n, n, tip) for _, n, tip in scored]
    return full or _FALLBACK


def _best_model_for(names: list[str], role: str) -> str | None:
    """Return the highest-ranked available model name for 'gen' or 'ana'."""
    key = "gen" if role == "gen" else "ana"
    scored = []
    for n in names:
        if _is_embed_model(n):
            continue
        profile = _get_model_profile(n)
        scored.append((profile[key] if profile else 99, n))
    scored.sort()
    return scored[0][1] if scored else None


def _rebuild_model_lists(all_names: list[str]) -> None:
    """Populate both role-specific item lists from a fresh set of model names."""
    global _model_items_generate, _model_items_analyze
    _model_items_generate = _build_model_items_for_role(all_names, "gen")
    _model_items_analyze = _build_model_items_for_role(all_names, "ana")


def _enum_generate_items(self, context):
    return _model_items_generate


def _enum_analyze_items(self, context):
    return _model_items_analyze


def _on_mode_change(self, context):
    """When mode switches, reset the model slot to the best for the new role."""
    if self.qna_mode == "GENERATE":
        available = [item[0] for item in _model_items_generate if item[0] != "none"]
        best = _best_model_for(available, "gen")
        if best:
            self.qna_model_generate = best
    else:
        available = [item[0] for item in _model_items_analyze if item[0] != "none"]
        best = _best_model_for(available, "ana")
        if best:
            self.qna_model_analyze = best


# === NODE TREE SERIALIZER ===


def _socket_default(socket) -> object:
    if not hasattr(socket, "default_value"):
        return None
    val = socket.default_value
    if hasattr(val, "__iter__") and not isinstance(val, str):
        try:
            return [round(v, 5) for v in val]
        except (TypeError, ValueError):
            return str(val)
    if isinstance(val, float):
        return round(val, 5)
    if isinstance(val, (int, bool, str)):
        return val
    return str(val)


def _serialize_val(val):
    """Serialize a Blender property value to JSON-safe types."""
    if isinstance(val, bpy.types.ID):
        return {"__datablock__": True, "id_type": type(val).__name__, "name": val.name}
    elif (
        "Vector" in str(type(val))
        or "Color" in str(type(val))
        or "Euler" in str(type(val))
    ):
        return list(val)
    elif type(val) in (int, float, str, bool, type(None)):
        return val
    elif isinstance(val, set):
        return list(val)
    elif hasattr(val, "name") and type(val).__name__ != "type":
        return str(val.name)
    return None


def _deserialize_val(val):
    """Deserialize a JSON value back to a Blender property value."""
    if isinstance(val, dict) and val.get("__datablock__"):
        id_type = val.get("id_type")
        name = val.get("name")
        if id_type == "Material" and name in bpy.data.materials:
            return bpy.data.materials[name]
        elif id_type == "Object" and name in bpy.data.objects:
            return bpy.data.objects[name]
        elif id_type == "Image" and name in bpy.data.images:
            return bpy.data.images[name]
        elif id_type == "Collection" and name in bpy.data.collections:
            return bpy.data.collections[name]
        elif id_type == "Texture" and name in bpy.data.textures:
            return bpy.data.textures[name]
        elif "NodeTree" in (id_type or "") and name in bpy.data.node_groups:
            return bpy.data.node_groups[name]
        return None
    return val


def serialize_node_tree(
    node_tree: bpy.types.NodeTree, _shared_groups: dict | None = None
) -> dict:
    """Recursively serialise *node_tree* and all sub-node-groups it references.

    Follows the same structured import/export pattern as Up3date's CityJSON
    parser/exporter: interface, nodes with full properties, links, and
    embedded sub-groups are all captured with identifiers for reliable
    round-trip reconstruction.

    *_shared_groups* is a flat dict ``{group_name: data}`` shared across
    recursive calls so each sub-group is only exported once.
    """
    is_root = _shared_groups is None
    if is_root:
        _shared_groups = {}

    data: dict = OrderedDict()
    data["tree_name"] = node_tree.name
    data["tree_type"] = node_tree.bl_idname

    # Extract interface (Group Inputs/Outputs)
    data["interface"] = []
    if hasattr(node_tree, "interface"):
        for item in getattr(node_tree.interface, "items_tree", []):
            if getattr(item, "item_type", "SOCKET") == "SOCKET":
                data["interface"].append(
                    {
                        "name": item.name,
                        "in_out": item.in_out,
                        "socket_type": item.socket_type,
                        "identifier": getattr(item, "identifier", ""),
                    }
                )

    nodes_list = []
    for node in node_tree.nodes:
        nd: dict = OrderedDict()
        nd["type"] = node.type
        nd["bl_idname"] = node.bl_idname
        nd["name"] = node.name
        nd["label"] = node.label or ""
        nd["location"] = [round(node.location.x, 1), round(node.location.y, 1)]
        nd["width"] = getattr(node, "width", 140.0)
        nd["muted"] = node.mute
        nd["use_custom_color"] = getattr(node, "use_custom_color", False)
        nd["color"] = list(node.color)

        # Serialize node properties via bl_rna (same pattern as json_mover.py / Up3date)
        props: dict = {}
        skip_keys = {
            "name",
            "type",
            "location",
            "dimensions",
            "width",
            "height",
            "inputs",
            "outputs",
            "bl_idname",
            "bl_label",
            "bl_description",
            "bl_icon",
            "bl_width_default",
            "bl_width_min",
            "bl_width_max",
            "bl_height_default",
            "bl_height_min",
            "bl_height_max",
            "location_absolute",
            "warning_propagation",
            "select",
            "hide",
            "mute",
            "label",
            "active_item",
        }
        if hasattr(node, "node_tree") and node.node_tree:
            sub_ng = node.node_tree
            s_val = _serialize_val(sub_ng)
            if s_val is not None:
                props["node_tree"] = s_val
            # Recursively embed the sub-group
            if sub_ng.name not in _shared_groups:
                _shared_groups[sub_ng.name] = {}
                sub_type = type(sub_ng).__name__
                _shared_groups[sub_ng.name] = serialize_node_tree(
                    sub_ng, _shared_groups
                )

        for key in node.bl_rna.properties.keys():
            if key in skip_keys:
                continue
            prop = node.bl_rna.properties[key]
            if not prop.is_readonly:
                try:
                    val = getattr(node, key)
                    s_val = _serialize_val(val)
                    if s_val is not None:
                        props[key] = s_val
                except Exception:
                    pass
        if props:
            nd["properties"] = props

        # Socket defaults with identifiers
        nd["inputs"] = []
        for idx, s in enumerate(node.inputs):
            sd: dict = OrderedDict()
            sd["name"] = s.name
            sd["type"] = s.bl_idname
            sd["identifier"] = getattr(s, "identifier", "")
            sd["default_value"] = _socket_default(s)
            nd["inputs"].append(sd)

        nd["outputs"] = []
        for idx, s in enumerate(node.outputs):
            sd = OrderedDict()
            sd["name"] = s.name
            sd["type"] = s.bl_idname
            sd["identifier"] = getattr(s, "identifier", "")
            sd["default_value"] = _socket_default(s)
            nd["outputs"].append(sd)

        # Parent frame reference
        if hasattr(node, "parent") and node.parent:
            nd["parent"] = node.parent.name

        # Dynamic node items (Capture Attribute, Store Named Attribute, Repeat Zones, Simulation Zones)
        items_collections = [
            "capture_items",
            "store_items",
            "repeat_items",
            "simulation_items",
        ]
        node_items = {}
        for col_name in items_collections:
            if hasattr(node, col_name):
                col = getattr(node, col_name)
                extracted_col = []
                for item in col:
                    item_data = {}
                    for k in item.bl_rna.properties.keys():
                        if k not in ("rna_type", "name"):
                            try:
                                v = getattr(item, k)
                                sv = _serialize_val(v)
                                if sv is not None:
                                    item_data[k] = sv
                            except Exception:
                                pass
                    extracted_col.append(item_data)
                if extracted_col:
                    node_items[col_name] = extracted_col
        if node_items:
            nd["node_items"] = node_items

        nodes_list.append(nd)
    data["nodes"] = nodes_list

    # Links with identifiers and indices
    links_list = []
    for link in node_tree.links:
        try:
            from_idx = list(link.from_node.outputs).index(link.from_socket)
        except ValueError:
            from_idx = None
        try:
            to_idx = list(link.to_node.inputs).index(link.to_socket)
        except ValueError:
            to_idx = None
        ld: dict = OrderedDict()
        ld["from_node"] = link.from_node.name
        ld["from_socket"] = getattr(
            link.from_socket, "identifier", link.from_socket.name
        )
        ld["from_socket_name"] = link.from_socket.name
        ld["from_socket_index"] = from_idx
        ld["to_node"] = link.to_node.name
        ld["to_socket"] = getattr(link.to_socket, "identifier", link.to_socket.name)
        ld["to_socket_name"] = link.to_socket.name
        ld["to_socket_index"] = to_idx
        ld["is_muted"] = link.is_muted
        links_list.append(ld)
    data["links"] = links_list

    if is_root:
        data["embedded_groups"] = _shared_groups

    return data


# === NODE REFERENCE (exact socket names for Blender 5.0) ===

SHADER_NODE_REFERENCE = """
=== BLENDER 5.0 SHADER NODE REFERENCE ===
Use ONLY exact bl_idname and socket names from this reference.

--- ShaderNodeOutputMaterial ---
  inputs: Surface(Shader), Volume(Shader), Displacement(Vector), Thickness(Float)

--- ShaderNodeBsdfPrincipled ---
  inputs: Base Color(RGBA), Metallic(Float), Roughness(Float=0.5),
          IOR(Float=1.5), Alpha(Float=1), Normal(Vector),
          Subsurface Weight(Float), Specular IOR Level(Float=0.5),
          Anisotropic(Float), Anisotropic Rotation(Float), Tangent(Vector),
          Transmission Weight(Float), Coat Weight(Float), Coat Roughness(Float),
          Coat IOR(Float=1.5), Coat Tint(RGBA), Coat Normal(Vector),
          Sheen Weight(Float), Sheen Roughness(Float=0.5), Sheen Tint(RGBA),
          Emission Color(RGBA), Emission Strength(Float=0)
  outputs: BSDF(Shader)

--- ShaderNodeBsdfDiffuse ---
  inputs: Color(RGBA), Roughness(Float), Normal(Vector)
  outputs: BSDF(Shader)

--- ShaderNodeBsdfGlossy ---
  inputs: Color(RGBA), Roughness(Float), Normal(Vector)
  outputs: BSDF(Shader)
  properties: distribution = "MULTI_GGX"|"GGX"|"BECKMANN"|"ASHIKHMIN_SHIRLEY"

--- ShaderNodeBsdfGlass ---
  inputs: Color(RGBA), Roughness(Float), IOR(Float=1.45), Normal(Vector)
  outputs: BSDF(Shader)

--- ShaderNodeBsdfTransparent ---
  inputs: Color(RGBA)
  outputs: BSDF(Shader)

--- ShaderNodeEmission ---
  inputs: Color(RGBA), Strength(Float=1)
  outputs: Emission(Shader)

--- ShaderNodeMixShader ---
  inputs: Factor(Float=0.5), Shader(Shader), Shader(Shader)
  outputs: Shader(Shader)

--- ShaderNodeAddShader ---
  inputs: Shader(Shader), Shader(Shader)
  outputs: Shader(Shader)

--- ShaderNodeTexNoise ---
  inputs: Vector(Vector), W(Float), Scale(Float=5), Detail(Float=2),
          Roughness(Float=0.5), Lacunarity(Float=2), Distortion(Float=0)
  outputs: Factor(Float), Color(RGBA)

--- ShaderNodeTexVoronoi ---
  inputs: Vector(Vector), W(Float), Scale(Float=5), Detail(Float=0),
          Roughness(Float=0.5), Lacunarity(Float=2), Randomness(Float=1)
  outputs: Distance(Float), Color(RGBA), Position(Vector), W(Float)
  properties: feature = "F1"|"F2"|"SMOOTH_F1"|"DISTANCE_TO_EDGE"|"N_SPHERE_RADIUS"
             distance = "EUCLIDEAN"|"MANHATTAN"|"CHEBYCHEV"|"MINKOWSKI"

--- ShaderNodeTexWave ---
  inputs: Vector(Vector), Scale(Float=5), Distortion(Float=0),
          Detail(Float=2), Detail Scale(Float=1), Detail Roughness(Float=0.5), Phase Offset(Float=0)
  outputs: Color(RGBA), Factor(Float)
  properties: wave_type = "BANDS"|"RINGS"

--- ShaderNodeTexChecker ---
  inputs: Vector(Vector), Color1(RGBA), Color2(RGBA), Scale(Float=5)
  outputs: Color(RGBA), Factor(Float)

--- ShaderNodeTexBrick ---
  inputs: Vector(Vector), Color1(RGBA), Color2(RGBA), Mortar(RGBA),
          Scale(Float=5), Mortar Size(Float=0.02), Mortar Smooth(Float=0.1),
          Bias(Float=0), Brick Width(Float=0.5), Row Height(Float=0.25)
  outputs: Color(RGBA), Factor(Float)

--- ShaderNodeTexGradient ---
  inputs: Vector(Vector)
  outputs: Color(RGBA), Factor(Float)
  properties: gradient_type = "LINEAR"|"QUADRATIC"|"EASING"|"DIAGONAL"|"SPHERICAL"|"RADIAL"

--- ShaderNodeTexMagic ---
  inputs: Vector(Vector), Scale(Float=5), Distortion(Float=1)
  outputs: Color(RGBA), Factor(Float)

--- ShaderNodeTexImage ---
  inputs: Vector(Vector)
  outputs: Color(RGBA), Alpha(Float)

--- ShaderNodeTexWhiteNoise ---
  inputs: Vector(Vector), W(Float)
  outputs: Value(Float), Color(RGBA)

--- ShaderNodeMix ---
  inputs: Factor(Float=0.5), A(varies), B(varies)
  outputs: Result(varies)
  properties: data_type = "FLOAT"|"VECTOR"|"RGBA"
             blend_type = "MIX"|"ADD"|"MULTIPLY"|"SUBTRACT"|"SCREEN"|"OVERLAY"|"DARKEN"|"LIGHTEN"

--- ShaderNodeInvert ---
  inputs: Factor(Float=1), Color(RGBA)
  outputs: Color(RGBA)

--- ShaderNodeHueSaturation ---
  inputs: Hue(Float=0.5), Saturation(Float=1), Value(Float=1), Factor(Float=1), Color(RGBA)
  outputs: Color(RGBA)

--- ShaderNodeBrightContrast ---
  inputs: Color(RGBA), Bright(Float=0), Contrast(Float=0)
  outputs: Color(RGBA)

--- ShaderNodeGamma ---
  inputs: Color(RGBA), Gamma(Float=1)
  outputs: Color(RGBA)

--- ShaderNodeBump ---
  inputs: Strength(Float=1), Distance(Float=1), Height(Float), Normal(Vector)
  outputs: Normal(Vector)

--- ShaderNodeNormalMap ---
  inputs: Strength(Float=1), Color(RGBA)
  outputs: Normal(Vector)

--- ShaderNodeDisplacement ---
  inputs: Height(Float), Midlevel(Float=0.5), Scale(Float=1), Normal(Vector)
  outputs: Displacement(Vector)

--- ShaderNodeMapping ---
  inputs: Vector(Vector), Location(Vector), Rotation(Vector), Scale(Vector)
  outputs: Vector(Vector)

--- ShaderNodeVectorMath ---
  inputs: Vector(Vector), Vector(Vector), Scale(Float=1)
  outputs: Vector(Vector), Value(Float)
  properties: operation = "ADD"|"SUBTRACT"|"MULTIPLY"|"NORMALIZE"|"SCALE"|"CROSS_PRODUCT"|"DOT_PRODUCT"

--- ShaderNodeMath ---
  inputs: Value(Float), Value(Float), Value(Float)
  outputs: Value(Float)
  properties: operation = "ADD"|"SUBTRACT"|"MULTIPLY"|"DIVIDE"|"POWER"|"SQRT"|"ABSOLUTE"|
              "MINIMUM"|"MAXIMUM"|"LESS_THAN"|"GREATER_THAN"|"ROUND"|"FLOOR"|"CEIL"|"FRACT"|
              "MODULO"|"SINE"|"COSINE"|"TANGENT"

--- ShaderNodeMapRange ---
  inputs: Value(Float), From Min(Float=0), From Max(Float=1), To Min(Float=0), To Max(Float=1)
  outputs: Result(Float)

--- ShaderNodeClamp ---
  inputs: Value(Float), Min(Float=0), Max(Float=1)
  outputs: Result(Float)

--- ShaderNodeValToRGB (ColorRamp) ---
  inputs: Factor(Float=0.5)
  outputs: Color(RGBA), Alpha(Float)

--- ShaderNodeRGBToBW ---
  inputs: Color(RGBA)
  outputs: Val(Float)

--- ShaderNodeCombineXYZ ---
  inputs: X(Float), Y(Float), Z(Float)
  outputs: Vector(Vector)

--- ShaderNodeSeparateXYZ ---
  inputs: Vector(Vector)
  outputs: X(Float), Y(Float), Z(Float)

--- ShaderNodeCombineColor ---
  inputs: Red(Float), Green(Float), Blue(Float)
  outputs: Color(RGBA)

--- ShaderNodeSeparateColor ---
  inputs: Color(RGBA)
  outputs: Red(Float), Green(Float), Blue(Float)

--- ShaderNodeTexCoord ---
  outputs: Generated(Vector), Normal(Vector), UV(Vector), Object(Vector),
           Camera(Vector), Window(Vector), Reflection(Vector)

--- ShaderNodeValue ---
  outputs: Value(Float)

--- ShaderNodeRGB ---
  outputs: Color(RGBA)

--- ShaderNodeFresnel ---
  inputs: IOR(Float=1.45), Normal(Vector)
  outputs: Factor(Float)

--- ShaderNodeLayerWeight ---
  inputs: Blend(Float=0.5), Normal(Vector)
  outputs: Fresnel(Float), Facing(Float)

--- ShaderNodeObjectInfo ---
  outputs: Location(Vector), Color(RGBA), Alpha(Float), Object Index(Float), Random(Float)

--- ShaderNodeGeometry ---
  outputs: Position(Vector), Normal(Vector), Tangent(Vector), True Normal(Vector),
           Incoming(Vector), Parametric(Vector), Backfacing(Float), Pointiness(Float)

--- ShaderNodeAmbientOcclusion ---
  inputs: Color(RGBA), Distance(Float=1), Normal(Vector)
  outputs: Color(RGBA), AO(Float)

--- ShaderNodeWireframe ---
  inputs: Size(Float=0.01)
  outputs: Factor(Float)

--- ShaderNodeShaderToRGB ---
  inputs: Shader(Shader)
  outputs: Color(RGBA), Alpha(Float)
"""

GEOMETRY_NODE_REFERENCE = """
=== BLENDER 5.0 GEOMETRY NODE REFERENCE ===

--- NodeGroupOutput ---
  inputs: Geometry(Geometry)
--- NodeGroupInput ---
  outputs: Geometry(Geometry)

--- Mesh Primitives ---
GeometryNodeMeshCube: outputs Mesh(Geometry). inputs: Size(Vector)
GeometryNodeMeshGrid: outputs Mesh(Geometry). inputs: Size X(Float), Size Y(Float), Vertices X(Int), Vertices Y(Int)
GeometryNodeMeshUVSphere: outputs Mesh(Geometry). inputs: Segments(Int), Rings(Int), Radius(Float)
GeometryNodeMeshIcoSphere: outputs Mesh(Geometry). inputs: Radius(Float), Subdivisions(Int)
GeometryNodeMeshCylinder: outputs Mesh(Geometry). inputs: Vertices(Int), Radius(Float), Depth(Float)
GeometryNodeMeshCone: outputs Mesh(Geometry). inputs: Vertices(Int), Radius Top(Float), Radius Bottom(Float), Depth(Float)
GeometryNodeMeshCircle: outputs Mesh(Geometry). inputs: Vertices(Int), Radius(Float)
GeometryNodeMeshLine: outputs Mesh(Geometry). inputs: Count(Int), Start Location(Vector), Offset(Vector)
GeometryNodeMeshTriangle: outputs Mesh(Geometry). inputs: Size(Vector)
GeometryNodeCurvePrimitiveLine: outputs Curve(Geometry). inputs: Start(Vector), End(Vector)
GeometryNodeCurvePrimitiveCircle: outputs Curve(Geometry). inputs: Mode(Enum), Radius(Float), Resolution(Int)
GeometryNodeCurvePrimitiveQuadrilateral: outputs Curve(Geometry). inputs: Mode(Enum), Start(Vector), Side 1(Vector), Side 2(Vector)
GeometryNodeCurvePrimitiveRectangle: outputs Curve(Geometry). inputs: Mode(Enum), Width(Float), Height(Float), Corners Radius(Float)

--- Geometry Operations ---
GeometryNodeJoinGeometry: inputs Geometry(Geometry). outputs Geometry(Geometry)
GeometryNodeTransform: inputs Geometry(Geometry), Translation(Vector), Rotation(Vector), Scale(Vector). outputs Geometry(Geometry)
GeometryNodeSetPosition: inputs Geometry(Geometry), Position(Vector), Offset(Vector), Selection(Bool). outputs Geometry(Geometry)
GeometryNodeSetMaterial: inputs Geometry(Geometry), Material(Material), Selection(Bool). outputs Geometry(Geometry)
GeometryNodeSubdivisionSurface: inputs Geometry(Geometry), Level(Int). outputs Geometry(Geometry)
GeometryNodeBoolean: inputs Mesh 1(Geometry), Mesh 2(Geometry), Solver(Enum). outputs Mesh(Geometry). properties: operation = "UNION"|"DIFFERENCE"|"INTERSECT"
GeometryNodeExtrudeMesh: inputs Mesh(Geometry), Selection(Bool), Offset(Float), Thickness(Float), Individual(Bool). outputs Mesh(Geometry)
GeometryNodeDeleteGeometry: inputs Geometry(Geometry), Selection(Bool). outputs Geometry(Geometry). properties: domain = "AUTO"|"POINT"|"EDGE"|"FACE"|"INSTANCE"
GeometryNodeSplitToInstances: inputs Geometry(Geometry), Selection(Bool). outputs Instances(Geometry). properties: domain = "AUTO"|"POINT"|"EDGE"|"FACE"
GeometryNodeMergeByDistance: inputs Geometry(Geometry), Selection(Bool), Distance(Float). outputs Geometry(Geometry)
GeometryNodeSeparateGeometry: inputs Geometry(Geometry), Selection(Bool). outputs Geometry(Geometry), Inverted(Geometry). properties: domain = "AUTO"|"POINT"|"EDGE"|"FACE"|"INSTANCE"
GeometryNodeDualMesh: inputs Mesh(Geometry). outputs Mesh(Geometry)
GeometryNodeTriangulate: inputs Mesh(Geometry), Selection(Bool). outputs Mesh(Geometry)
GeometryNodeConvexHull: inputs Geometry(Geometry), Position(Vector). outputs Geometry(Geometry), Vertices(Geometry), Edges(Geometry), Triangles(Geometry)
GeometryNodeBoundingBox: inputs Geometry(Geometry). outputs Bounding Box(Geometry), Min(Vector), Max(Vector), Size(Vector), Center(Vector)

--- Curve Operations ---
GeometryNodeCurveToMesh: inputs Curve(Geometry), Profile Curve(Geometry), Fill Caps(Bool). outputs Mesh(Geometry)
GeometryNodeCurveToPoints: inputs Curve(Geometry), Mode(Enum). outputs Points(Geometry). properties: mode = "EVALUATED"|"CONTROL_POINTS"
GeometryNodePointsToCurves: inputs Points(Geometry). outputs Curves(Geometry)
GeometryNodeResampleCurve: inputs Curve(Geometry), Mode(Enum), Count(Int), Length(Float). outputs Curve(Geometry). properties: mode = "COUNT"|"LENGTH"
GeometryNodeSubdivideCurve: inputs Curve(Geometry), Cuts(Int). outputs Curve(Geometry)
GeometryNodeTrimCurve: inputs Curve(Geometry), Mode(Enum), Start(Float), End(Float), Length(Float). outputs Curve(Geometry). properties: mode = "FACTOR"|"LENGTH"
GeometryNodeFilletCurve: inputs Curve(Geometry), Selection(Bool), Radius(Float), Segments(Int). outputs Curve(Geometry)
GeometryNodeInterpolateCurves: inputs Curve 1(Geometry), Curve 2(Geometry), Factor(Float). outputs Curve(Geometry)
GeometryNodeOffsetPointInCurve: inputs Curve(Geometry), Factor(Float), Offset(Float). outputs Vector(Vector)
GeometryNodeSplineParameter: inputs Spline(Geometry), Factor(Float). outputs Position(Vector), Tangent(Vector), Normal(Vector), Curvature(Float)
GeometryNodeSplineType: inputs Curve(Geometry). outputs Curve(Geometry). properties: type = "POLY"|"BEZIER"|"NURBS"|"CARDINAL"
GeometryNodeSetCurveRadius: inputs Curve(Geometry), Value(Float), Selection(Bool). outputs Curve(Geometry)
GeometryNodeSetCurveTilt: inputs Curve(Geometry), Value(Float), Selection(Bool). outputs Curve(Geometry)
GeometryNodeSetHandleType: inputs Curve(Geometry), Handle Type(Enum), Selection(Bool). outputs Curve(Geometry). properties: handle_type = "AUTO"|"VECTOR"|"ALIGN"|"FREE_ALIGN"
GeometryNodeSetSplineResolution: inputs Curve(Geometry), Resolution(Int), Selection(Bool). outputs Curve(Geometry)
GeometryNodeReverseCurve: inputs Curve(Geometry), Selection(Bool). outputs Curve(Geometry)

--- Instances ---
GeometryNodeInstanceOnPoints: inputs Points(Geometry), Instance(Geometry), Selection(Bool), Pick Instance(Bool), Instance Index(Int), Position(Vector), Rotation(Vector), Scale(Vector), Random ID(Int). outputs Instances(Geometry)
GeometryNodeRealizeInstances: inputs Geometry(Geometry). outputs Geometry(Geometry)
GeometryNodeInstancesToPoints: inputs Instances(Geometry). outputs Points(Geometry)
GeometryNodeImageInfo: outputs Width(Int), Height(Int), Color Space(Enum), Has Alpha(Bool), Frame Count(Int), Is Animated(Bool)
GeometryNodeImageTexture: inputs Vector(Vector), Image(Image). outputs Color(Color), Alpha(Float)

--- Points ---
GeometryNodeDistributePointsOnFaces: inputs Mesh(Geometry), Selection(Bool), Density(Float), Density Max(Float), Seed(Int). outputs Points(Geometry). properties: distribute_method = "RANDOM"|"POISSON_DISK"
GeometryNodePointsToVertices: inputs Points(Geometry). outputs Mesh(Geometry)

--- Inputs ---
GeometryNodeInputPosition: outputs Position(Vector)
GeometryNodeInputNormal: outputs Normal(Vector)
GeometryNodeInputIndex: outputs Index(Int)
GeometryNodeInputNamedAttribute: inputs Name(String). outputs Value(varies), Exists(Bool). properties: data_type = "AUTO"|"FLOAT"|"INT"|"FLOAT_VECTOR"|"FLOAT_COLOR"|"QUATERNION"
GeometryNodeInputSceneTime: outputs Time(Float), Frame(Float), Delta Time(Float)
GeometryNodeInputActiveCamera: outputs Camera(Object)
GeometryNodeObjectInfo: inputs Object(Object). outputs Geometry(Geometry), Transform(Matrix), Location(Vector), Rotation(Vector), Scale(Vector)
GeometryNodeCollectionInfo: inputs Collection(Collection), Reset Children(Bool). outputs Instances(Geometry)
GeometryNodeCameraInfo: outputs View Vector(Vector), View Z Depth(Float), View Distance(Float)
GeometryNodeIsViewport: outputs Is Viewport(Bool)

--- Attribute Operations ---
GeometryNodeStoreNamedAttribute: inputs Geometry(Geometry), Selection(Bool), Name(String), Value(varies). outputs Geometry(Geometry). properties: data_type = "FLOAT"|"INT"|"FLOAT_VECTOR"|"FLOAT_COLOR"|"QUATERNION", domain = "POINT"|"EDGE"|"FACE"|"CORNER"|"INSTANCE"
GeometryNodeCaptureAttribute: inputs Geometry(Geometry), Selection(Bool), Value(varies). outputs Geometry(Geometry), Captured(varies). properties: data_type = "FLOAT"|"INT"|"FLOAT_VECTOR"|"FLOAT_COLOR"|"QUATERNION", domain = "POINT"|"EDGE"|"FACE"|"CORNER"|"INSTANCE"
GeometryNodeNamedAttribute: inputs Name(String). outputs Value(varies), Exists(Bool). properties: data_type = "AUTO"|"FLOAT"|"INT"|"FLOAT_VECTOR"|"FLOAT_COLOR"|"QUATERNION"
GeometryNodeRemoveAttribute: inputs Geometry(Geometry), Selection(Bool), Name(String). outputs Geometry(Geometry). properties: domain = "POINT"|"EDGE"|"FACE"|"CORNER"|"INSTANCE"
GeometryNodeFieldAtIndex: inputs Value(varies), Index(Int). outputs Value(varies). properties: data_type = "FLOAT"|"INT"|"FLOAT_VECTOR"|"FLOAT_COLOR"|"QUATERNION", domain = "POINT"|"EDGE"|"FACE"|"CORNER"|"INSTANCE"
GeometryNodeDomainSize: inputs Geometry(Geometry). outputs Size(Int). properties: domain = "POINT"|"EDGE"|"FACE"|"CORNER"|"INSTANCE"|"SPLINE"|"CURVE"

--- Simulation / Repeat ---
GeometryNodeSimulationInput: inputs Geometry(Geometry). outputs Geometry(Geometry)
GeometryNodeSimulationOutput: outputs Geometry(Geometry)
GeometryNodeRepeatInput: outputs Geometry(Geometry), Iteration Number(Int)
GeometryNodeRepeatOutput: inputs Geometry(Geometry). outputs Geometry(Geometry)
GeometryNodeForEachElementInput: outputs Geometry(Geometry), Index(Int), Total Count(Int)
GeometryNodeForEachElementOutput: inputs Geometry(Geometry). outputs Geometry(Geometry)

--- Viewport Display ---
GeometryNodeViewer: inputs Geometry(Geometry), Value(varies). outputs Geometry(Geometry). properties: data_type = "FLOAT"|"INT"|"FLOAT_VECTOR"|"FLOAT_COLOR"|"QUATERNION", domain = "POINT"|"EDGE"|"FACE"|"CORNER"|"INSTANCE"

--- Matrix / Transform ---
FunctionNodeTransformPoint: inputs Transform(Matrix), Point(Vector). outputs Point(Vector)
FunctionNodeTransformDirection: inputs Transform(Matrix), Direction(Vector). outputs Direction(Vector)
FunctionNodeInvertMatrix: inputs Matrix(Matrix). outputs Inverse(Matrix)
FunctionNodeCombineMatrix: inputs Row 1(Vector), Row 2(Vector), Row 3(Vector), Row 4(Vector). outputs Matrix(Matrix)
FunctionNodeSeparateMatrix: inputs Matrix(Matrix). outputs Row 1(Vector), Row 2(Vector), Row 3(Vector), Row 4(Vector)
FunctionNodeRotateEuler: inputs Rotation(Vector), Rotate By(Vector), Type(Enum). outputs Rotation(Vector). properties: type = "XYZ"|"XZY"|"YXZ"|"YZX"|"ZXY"|"ZYX"|"QUATERNION_AXIS_ANGLE"
FunctionNodeAlignEulerToVector: inputs Vector(Vector), Axis(Enum), Pivot Axis(Enum). outputs Rotation(Vector). properties: axis = "X"|"Y"|"Z", pivot_axis = "X"|"Y"|"Z"
FunctionNodeProjectPoint: inputs Point(Vector), Start Point(Vector), End Point(Vector). outputs Result(Vector), Factor(Float), Distance(Float)

--- Math (shared with Shader) ---
ShaderNodeMath: inputs Value(Float), Value(Float). outputs Value(Float). properties: operation
ShaderNodeVectorMath: inputs Vector(Vector), Vector(Vector). outputs Vector(Vector), Value(Float). properties: operation
ShaderNodeCombineXYZ: inputs X(Float), Y(Float), Z(Float). outputs Vector(Vector)
ShaderNodeSeparateXYZ: inputs Vector(Vector). outputs X(Float), Y(Float), Z(Float)
FunctionNodeRandomValue: outputs Value(Float). properties: data_type = "FLOAT"|"INT"|"FLOAT_VECTOR"|"BOOLEAN"
FunctionNodeCompare: inputs A(Float), B(Float). outputs Result(Boolean). properties: data_type, operation
"""


# === SYSTEM PROMPTS ===

_ANALYZE_SYSTEM = (
    "You are a Blender node-graph expert. The user will provide the full "
    "JSON representation of a Blender node tree (shader or geometry nodes) "
    "and ask a question about it. Analyze the node graph thoroughly. "
    "Reference nodes by their name/label. Be concise but precise."
)

# Step 1: Node selection prompt
_FEW_SHOT_SHADER_STEP1 = """
EXAMPLES (verified working setups — follow this exact schema):

User: Prism glass with red, green, blue wavelength layers
{"nodes": [
  {"id": "glass_r", "bl_idname": "ShaderNodeBsdfGlass", "inputs": {"Color": [1,0,0,1], "Roughness": 0.0, "IOR": 1.4}, "properties": {"distribution": "BECKMANN"}},
  {"id": "glass_g", "bl_idname": "ShaderNodeBsdfGlass", "inputs": {"Color": [0,1,0,1], "Roughness": 0.0, "IOR": 1.45}, "properties": {"distribution": "BECKMANN"}},
  {"id": "glass_b", "bl_idname": "ShaderNodeBsdfGlass", "inputs": {"Color": [0,0,1,1], "Roughness": 0.0, "IOR": 1.5}, "properties": {"distribution": "BECKMANN"}},
  {"id": "add1", "bl_idname": "ShaderNodeAddShader"},
  {"id": "add2", "bl_idname": "ShaderNodeAddShader"},
  {"id": "add3", "bl_idname": "ShaderNodeAddShader"},
  {"id": "mix", "bl_idname": "ShaderNodeMixShader"},
  {"id": "output", "bl_idname": "ShaderNodeOutputMaterial"}
]}

User: Shiny blue glossy plastic with Fresnel rim reflection
{"nodes": [
  {"id": "diffuse", "bl_idname": "ShaderNodeBsdfDiffuse", "inputs": {"Color": [0.04, 0.78, 0.8, 1.0], "Roughness": 0.001}},
  {"id": "glossy", "bl_idname": "ShaderNodeBsdfAnisotropic", "inputs": {"Roughness": 0.1}, "properties": {"distribution": "GGX"}},
  {"id": "glossy2", "bl_idname": "ShaderNodeBsdfAnisotropic", "inputs": {"Roughness": 0.1}, "properties": {"distribution": "GGX"}},
  {"id": "fresnel", "bl_idname": "ShaderNodeFresnel", "inputs": {"IOR": 1.45}},
  {"id": "mix2", "bl_idname": "ShaderNodeMixShader", "inputs": {"Factor": 0.05}},
  {"id": "mix", "bl_idname": "ShaderNodeMixShader"},
  {"id": "output", "bl_idname": "ShaderNodeOutputMaterial"}
]}

User: Ghibli-style grass shader with vertex attribute color, noise variation, and camera-ray transparency
{"nodes": [
  {"id": "attr_normal", "bl_idname": "ShaderNodeAttribute"},
  {"id": "attr_pos",    "bl_idname": "ShaderNodeAttribute"},
  {"id": "attr_uv",     "bl_idname": "ShaderNodeAttribute"},
  {"id": "sep_xyz",     "bl_idname": "ShaderNodeSeparateXYZ"},
  {"id": "fcurve_x",    "bl_idname": "ShaderNodeFloatCurve"},
  {"id": "fcurve_y",    "bl_idname": "ShaderNodeFloatCurve"},
  {"id": "mix_color",   "bl_idname": "ShaderNodeMix", "properties": {"blend_type": "BURN", "data_type": "RGBA"}},
  {"id": "math_alpha",  "bl_idname": "ShaderNodeMath", "properties": {"operation": "GREATER_THAN"}},
  {"id": "noise_tex",   "bl_idname": "ShaderNodeTexNoise", "inputs": {"Scale": 0.3, "Detail": 2.0}},
  {"id": "bright_con",  "bl_idname": "ShaderNodeBrightContrast", "inputs": {"Contrast": 0.7}},
  {"id": "color_ramp",  "bl_idname": "ShaderNodeValToRGB"},
  {"id": "diffuse",     "bl_idname": "ShaderNodeBsdfDiffuse"},
  {"id": "transparent", "bl_idname": "ShaderNodeBsdfTransparent"},
  {"id": "transparent2","bl_idname": "ShaderNodeBsdfTransparent"},
  {"id": "mix_shader",  "bl_idname": "ShaderNodeMixShader"},
  {"id": "mix_shader2", "bl_idname": "ShaderNodeMixShader"},
  {"id": "light_path",  "bl_idname": "ShaderNodeLightPath"},
  {"id": "output",      "bl_idname": "ShaderNodeOutputMaterial"}
]}
"""

_STEP1_SYSTEM_SHADER = (
    """You are a Blender 5.0 shader node expert.
Given a material description, output a JSON object listing the nodes needed.

RULES:
- Use ONLY bl_idname values from the reference below.
- Each node needs a unique "id" string (lowercase, underscores, e.g. "principled", "noise_tex").
- "properties" sets enum values like operation, blend_type, data_type.
- "inputs" sets default values by EXACT socket name. Colors are [R,G,B,A]. Vectors are [X,Y,Z].
- ALWAYS include ShaderNodeOutputMaterial with id "output".
- ALWAYS include a shader node (Principled, Diffuse, Glossy, etc.) that connects to output.
- Do NOT include links — only nodes. Links will be specified separately.
- Output raw JSON only. No explanation.

"""
    + SHADER_NODE_REFERENCE
    + _FEW_SHOT_SHADER_STEP1
)

_FEW_SHOT_GEO_STEP1 = """
EXAMPLE (verified working setup — follow this exact schema):

User: Distribute grass object collection on a surface with random scale and rotation
{"nodes": [
  {"id": "group_input", "bl_idname": "NodeGroupInput"},
  {"id": "group_output", "bl_idname": "NodeGroupOutput"},
  {"id": "join_geo", "bl_idname": "GeometryNodeJoinGeometry"},
  {"id": "dist_points", "bl_idname": "GeometryNodeDistributePointsOnFaces", "inputs": {"Density Max": 10.0, "Seed": 3}},
  {"id": "instance_on_pts", "bl_idname": "GeometryNodeInstanceOnPoints"},
  {"id": "collection_info", "bl_idname": "GeometryNodeCollectionInfo"},
  {"id": "math", "bl_idname": "ShaderNodeMath"},
  {"id": "math2", "bl_idname": "ShaderNodeMath", "properties": {"operation": "SUBTRACT"}},
  {"id": "vec_math", "bl_idname": "ShaderNodeVectorMath", "properties": {"operation": "SCALE"}},
  {"id": "rotate_euler", "bl_idname": "FunctionNodeRotateEuler"},
  {"id": "rand_scale", "bl_idname": "FunctionNodeRandomValue", "inputs": {"Min": 10.0, "Max": 12.0}, "properties": {"data_type": "FLOAT"}},
  {"id": "rand_rot", "bl_idname": "FunctionNodeRandomValue", "inputs": {"Max": [0,0,360]}, "properties": {"data_type": "FLOAT_VECTOR"}},
  {"id": "rand_noise", "bl_idname": "FunctionNodeRandomValue", "inputs": {"Max": [0,0,10]}, "properties": {"data_type": "FLOAT_VECTOR"}},
  {"id": "noise_tex", "bl_idname": "ShaderNodeTexNoise", "inputs": {"Scale": 5.0, "Detail": 2.0}}
]}
"""

_STEP1_SYSTEM_GEOMETRY = (
    """You are a Blender 5.0 geometry nodes expert.
Given a description of a procedural effect, output a JSON object listing the nodes needed.

RULES:
- Use ONLY bl_idname values from the reference below.
- Each node needs a unique "id" string.
- ALWAYS include NodeGroupInput (id "group_input") and NodeGroupOutput (id "group_output").
- ALWAYS include GeometryNodeJoinGeometry (id "join_geo") — it is REQUIRED in every setup.
  It acts as the final collector: every geometry stream feeds into it, then it connects to
  group_output. The group_input Geometry output also feeds into it as a pass-through.
- Do NOT include links — only nodes.
- Output raw JSON only.

OUTPUT TOPOLOGY RULE (mandatory):
  [any geometry producer] → join_geo → group_output
  group_input.Geometry   → join_geo  (always)

"""
    + GEOMETRY_NODE_REFERENCE
    + _FEW_SHOT_GEO_STEP1
)

# Step 2: Link connection prompt (socket info injected dynamically)
_STEP2_SYSTEM = """You are a Blender 5.0 node connection expert.
Given a list of nodes with their EXACT input and output socket names, specify how to connect them.

RULES:
- Use ONLY the exact node IDs and socket names provided below.
- from_socket must be an OUTPUT socket name. to_socket must be an INPUT socket name.
- Every shader setup must ultimately connect to the "Surface" input on the output node.
- Every geometry setup must route all geometry through join_geo → group_output "Geometry".
- Do not create circular connections.
- Output raw JSON only. No explanation.

SHADER EXAMPLES (verified working):

User: Connect prism glass nodes [ids: glass_r, glass_g, glass_b, add1, add2, add3, mix, output]
{"links": [
  {"from_node": "glass_r",  "from_socket": "BSDF",   "to_node": "add1",   "to_socket": "Shader"},
  {"from_node": "glass_g",  "from_socket": "BSDF",   "to_node": "add1",   "to_socket": "Shader"},
  {"from_node": "add1",     "from_socket": "Shader", "to_node": "add2",   "to_socket": "Shader"},
  {"from_node": "glass_b",  "from_socket": "BSDF",   "to_node": "add2",   "to_socket": "Shader"},
  {"from_node": "glass_b",  "from_socket": "BSDF",   "to_node": "add3",   "to_socket": "Shader"},
  {"from_node": "add2",     "from_socket": "Shader", "to_node": "mix",    "to_socket": "Shader"},
  {"from_node": "add3",     "from_socket": "Shader", "to_node": "mix",    "to_socket": "Shader"},
  {"from_node": "mix",      "from_socket": "Shader", "to_node": "output", "to_socket": "Surface"}
]}

User: Connect blue glossy plastic nodes [ids: diffuse, glossy, glossy2, fresnel, mix2, mix, output]
{"links": [
  {"from_node": "fresnel",  "from_socket": "Factor", "to_node": "mix",    "to_socket": "Factor"},
  {"from_node": "diffuse",  "from_socket": "BSDF",   "to_node": "mix2",   "to_socket": "Shader"},
  {"from_node": "glossy",   "from_socket": "BSDF",   "to_node": "mix2",   "to_socket": "Shader"},
  {"from_node": "glossy2",  "from_socket": "BSDF",   "to_node": "mix",    "to_socket": "Shader"},
  {"from_node": "mix2",     "from_socket": "Shader", "to_node": "mix",    "to_socket": "Shader"},
  {"from_node": "mix",      "from_socket": "Shader", "to_node": "output", "to_socket": "Surface"}
]}

User: Connect Ghibli grass shader nodes [ids: attr_normal, attr_pos, attr_uv, sep_xyz, fcurve_x, fcurve_y, mix_color, math_alpha, noise_tex, bright_con, color_ramp, diffuse, transparent, transparent2, mix_shader, mix_shader2, light_path, output]
{"links": [
  {"from_node": "attr_pos",     "from_socket": "Vector",        "to_node": "sep_xyz",     "to_socket": "Vector"},
  {"from_node": "sep_xyz",      "from_socket": "X",             "to_node": "fcurve_x",    "to_socket": "Value"},
  {"from_node": "sep_xyz",      "from_socket": "Y",             "to_node": "fcurve_y",    "to_socket": "Value"},
  {"from_node": "fcurve_x",     "from_socket": "Value",         "to_node": "mix_color",   "to_socket": "A"},
  {"from_node": "fcurve_y",     "from_socket": "Value",         "to_node": "mix_color",   "to_socket": "B"},
  {"from_node": "mix_color",    "from_socket": "Result",        "to_node": "math_alpha",  "to_socket": "Value"},
  {"from_node": "math_alpha",   "from_socket": "Value",         "to_node": "mix_shader",  "to_socket": "Factor"},
  {"from_node": "attr_uv",      "from_socket": "Vector",        "to_node": "noise_tex",   "to_socket": "Vector"},
  {"from_node": "noise_tex",    "from_socket": "Color",         "to_node": "bright_con",  "to_socket": "Color"},
  {"from_node": "bright_con",   "from_socket": "Color",         "to_node": "color_ramp",  "to_socket": "Factor"},
  {"from_node": "color_ramp",   "from_socket": "Color",         "to_node": "diffuse",     "to_socket": "Color"},
  {"from_node": "attr_normal",  "from_socket": "Vector",        "to_node": "diffuse",     "to_socket": "Normal"},
  {"from_node": "diffuse",      "from_socket": "BSDF",          "to_node": "mix_shader",  "to_socket": "Shader"},
  {"from_node": "transparent",  "from_socket": "BSDF",          "to_node": "mix_shader",  "to_socket": "Shader"},
  {"from_node": "mix_shader",   "from_socket": "Shader",        "to_node": "mix_shader2", "to_socket": "Shader"},
  {"from_node": "transparent2", "from_socket": "BSDF",          "to_node": "mix_shader2", "to_socket": "Shader"},
  {"from_node": "light_path",   "from_socket": "Is Camera Ray", "to_node": "mix_shader2", "to_socket": "Factor"},
  {"from_node": "mix_shader2",  "from_socket": "Shader",        "to_node": "output",      "to_socket": "Surface"}
]}

GEOMETRY EXAMPLE (verified working):

User: Connect grass distribution nodes [ids: group_input, group_output, join_geo, dist_points, instance_on_pts, collection_info, math, math2, vec_math, rotate_euler, rand_scale, rand_rot, rand_noise, noise_tex]
{"links": [
  {"from_node": "group_input",     "from_socket": "Geometry",   "to_node": "join_geo",        "to_socket": "Geometry"},
  {"from_node": "group_input",     "from_socket": "Geometry",   "to_node": "dist_points",     "to_socket": "Mesh"},
  {"from_node": "dist_points",     "from_socket": "Points",     "to_node": "instance_on_pts", "to_socket": "Points"},
  {"from_node": "collection_info", "from_socket": "Instances",  "to_node": "instance_on_pts", "to_socket": "Instance"},
  {"from_node": "math",            "from_socket": "Value",      "to_node": "dist_points",     "to_socket": "Density"},
  {"from_node": "rand_scale",      "from_socket": "Value",      "to_node": "instance_on_pts", "to_socket": "Scale"},
  {"from_node": "noise_tex",       "from_socket": "Color",      "to_node": "math2",           "to_socket": "Value"},
  {"from_node": "math2",           "from_socket": "Value",      "to_node": "vec_math",        "to_socket": "Vector"},
  {"from_node": "vec_math",        "from_socket": "Vector",     "to_node": "rotate_euler",    "to_socket": "Rotation"},
  {"from_node": "rand_noise",      "from_socket": "Value",      "to_node": "rotate_euler",    "to_socket": "Rotate By"},
  {"from_node": "rotate_euler",    "from_socket": "Rotation",   "to_node": "instance_on_pts", "to_socket": "Rotation"},
  {"from_node": "instance_on_pts", "from_socket": "Instances",  "to_node": "join_geo",        "to_socket": "Geometry"},
  {"from_node": "join_geo",        "from_socket": "Geometry",   "to_node": "group_output",    "to_socket": "Geometry"}
]}
"""


# === JSON SCHEMAS FOR STRUCTURED OUTPUT ===

_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "bl_idname": {"type": "string"},
                    "label": {"type": "string"},
                    "properties": {"type": "object"},
                    "inputs": {"type": "object"},
                },
                "required": ["id", "bl_idname"],
            },
        },
    },
    "required": ["nodes"],
}


def _build_link_schema(node_ids: list[str]) -> dict:
    """Build a JSON schema for links that constrains node IDs to valid values."""
    return {
        "type": "object",
        "properties": {
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from_node": {"type": "string", "enum": node_ids},
                        "from_socket": {"type": "string"},
                        "to_node": {"type": "string", "enum": node_ids},
                        "to_socket": {"type": "string"},
                    },
                    "required": ["from_node", "from_socket", "to_node", "to_socket"],
                },
            }
        },
        "required": ["links"],
    }


# === OLLAMA API ===


def _ollama_request_sync(
    base_url: str, model: str, messages: list[dict], format_schema: dict | None = None
) -> str:
    """
    Collects a full Ollama response as a single string.
    Uses stream=True internally so the abort flag is checked between tokens.
    If format_schema is provided, Ollama enforces the JSON structure.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,  # stream so we can abort between tokens
        "keep_alive": "5m",  # how long Ollama holds the model in VRAM after use (set "0" to unload immediately)
    }
    if format_schema:
        payload["format"] = format_schema

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    tokens: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = b""
        while True:
            if _state.abort_requested:
                break
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if chunk == b"\n":
                line = buf.strip()
                buf = b""
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                if token:
                    tokens.append(token)
                if obj.get("done", False):
                    break
    return "".join(tokens)


def _ollama_request_stream(base_url: str, model: str, messages: list[dict]):
    """Streaming POST to Ollama. Yields content tokens."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = b""
        while True:
            if _state.abort_requested:
                return
            chunk = resp.read(1)
            if not chunk:
                break
            buf += chunk
            if chunk == b"\n":
                line = buf.strip()
                buf = b""
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = obj.get("message", {}).get("content", "")
                if token:
                    yield token
                if obj.get("done", False):
                    return


# === BACKGROUND: ANALYZE (unchanged) ===


def _bg_analyze(base_url, model, messages):
    try:
        for token in _ollama_request_stream(base_url, model, messages):
            if _state.abort_requested:
                _state.raw_response += "\n[Aborted]"
                break
            _state.raw_response += token
            _state.dirty = True
    except urllib.error.URLError as exc:
        _state.error = f"Connection error: {exc.reason}"
        _state.dirty = True
    except Exception as exc:
        _state.error = f"Error: {exc}"
        _state.dirty = True
    finally:
        _state.is_running = False
        _state.dirty = True


# === NODE TREE IMPORTER (JSON → Blender) ===


def _resolve_socket_import(
    collection,
    old_id: str,
    name: str,
    index,
    interface_id_map: dict,
    group_id_maps: dict | None = None,
    group_node=None,
):
    """Return the best-matching socket from *collection*.

    Works correctly for both standard nodes and group boundary nodes
    (NodeGroupInput / NodeGroupOutput) whose identifiers change on every
    interface rebuild. Mirrors json_mover.py's _resolve_socket.
    """
    if (
        group_node is not None
        and group_id_maps
        and hasattr(group_node, "node_tree")
        and group_node.node_tree
    ):
        grp_map = group_id_maps.get(group_node.node_tree.name, {})
        mapped_id = grp_map.get(old_id, old_id)
        sock = collection.get(mapped_id)
        if sock:
            return sock

    mapped_id = interface_id_map.get(old_id, old_id)
    sock = collection.get(mapped_id)
    if sock:
        return sock

    if mapped_id != old_id:
        sock = collection.get(old_id)
        if sock:
            return sock

    name_matches = []
    for s in collection:
        if s.identifier == mapped_id or s.identifier == old_id:
            return s
        if s.name == name:
            name_matches.append(s)

    if len(name_matches) == 1:
        return name_matches[0]

    if index is not None and index < len(collection):
        return collection[index]

    if name_matches:
        return name_matches[0]

    return None


def _ensure_embedded_groups_import(embedded: dict) -> dict:
    """Recreate sub-node-groups embedded in an exported JSON.

    Two-pass strategy mirroring json_mover.py:
      Pass 1 — Create EMPTY SHELLS for groups that don't exist yet.
      Pass 2 — Build the contents of only the newly-created shells.

    Returns a dict ``{group_name: {old_id: new_id}}`` mapping exported interface
    identifiers to the Blender-generated ones.
    """
    group_id_maps = {}
    to_build = []
    for group_name, group_data in embedded.items():
        if not group_data:
            continue
        if group_name not in bpy.data.node_groups:
            group_type = group_data.get("tree_type", "GeometryNodeTree")
            bpy.data.node_groups.new(name=group_name, type=group_type)
            to_build.append(group_name)

    for group_name in to_build:
        ng = bpy.data.node_groups[group_name]
        id_map = _build_node_tree_from_json(ng, embedded[group_name])
        group_id_maps[group_name] = id_map

    return group_id_maps


def _build_node_tree_from_json(node_tree: bpy.types.NodeTree, data: dict) -> dict:
    """Build a node tree from a JSON data dict (the format produced by serialize_node_tree).

    This is the import counterpart to serialize_node_tree. It handles:
    - Interface recreation with identifier mapping
    - Embedded sub-node-groups (two-pass to avoid .001 duplicates)
    - Dynamic node items (capture_items, store_items, repeat_items, simulation_items)
    - Parent frame resolution (deferred to second-pass)
    - Socket resolution via identifiers with fallback chain
    - GeometryNodeGroup node_tree assignment retry in second-pass

    Returns interface_id_map mapping old identifiers to new ones.
    """
    node_tree.nodes.clear()
    created_nodes = {}

    # Ensure any embedded sub-groups exist before we try to assign node.node_tree
    embedded = data.get("embedded_groups", {})
    group_id_maps = {}
    if embedded:
        group_id_maps = _ensure_embedded_groups_import(embedded)

    # Recreate interface (needed for Group Input/Output nodes)
    interface_id_map = {}
    if hasattr(node_tree, "interface") and "interface" in data:
        node_tree.interface.clear()
        for idata in data["interface"]:
            try:
                new_sock = node_tree.interface.new_socket(
                    name=idata["name"],
                    in_out=idata["in_out"],
                    socket_type=idata["socket_type"],
                )
                if "identifier" in idata and idata["identifier"]:
                    interface_id_map[idata["identifier"]] = new_sock.identifier
            except Exception as e:
                print(f"[qna] Error creating interface socket {idata.get('name')}: {e}")

    # Create nodes
    for n_data in data.get("nodes", []):
        try:
            node = node_tree.nodes.new(n_data["bl_idname"])
        except RuntimeError:
            print(f"[qna] Node type {n_data['bl_idname']} not supported. Skipping.")
            continue

        node.name = n_data["name"]
        node.width = n_data.get("width", 140.0)

        if "use_custom_color" in n_data:
            node.use_custom_color = n_data["use_custom_color"]
        if "color" in n_data:
            node.color = n_data["color"]

        # Dynamic node items MUST be done before socket resolution
        for col_name, extracted_col in n_data.get("node_items", {}).items():
            if hasattr(node, col_name):
                col = getattr(node, col_name)
                col.clear()
                for item_data in extracted_col:
                    dt = item_data.get("data_type", "FLOAT")
                    name = item_data.get("name", "")
                    try:
                        new_item = col.new(dt, name)
                        for k, v in item_data.items():
                            if hasattr(new_item, k):
                                d_v = _deserialize_val(v)
                                if d_v is not None:
                                    try:
                                        setattr(new_item, k, d_v)
                                    except Exception:
                                        pass
                    except Exception as e:
                        print(f"[qna] Failed to create item in {col_name}: {e}")

        # Properties (defer parent and active_item to second-pass)
        for k, v in n_data.get("properties", {}).items():
            if k in ("parent", "active_item"):
                continue
            if hasattr(node, k):
                try:
                    deserialized_val = _deserialize_val(v)
                    if type(deserialized_val) == list and type(
                        getattr(node, k)
                    ).__name__ in ("Color", "Vector", "Euler"):
                        setattr(node, k, deserialized_val)
                    elif deserialized_val is not None:
                        setattr(node, k, deserialized_val)
                except Exception as e:
                    print(f"[qna] Could not set {k} on {node.name}: {e}")

        # Default socket values
        for sock_entry in n_data.get("inputs", []):
            sock_id = sock_entry.get("identifier", "")
            idx = sock_entry.get("index", 0)
            val = sock_entry.get("default_value")
            deserialized_val = _deserialize_val(val)

            target_sock = None
            if sock_id and sock_id in node.inputs:
                target_sock = node.inputs[sock_id]
            elif idx < len(node.inputs):
                target_sock = node.inputs[idx]

            if (
                target_sock
                and hasattr(target_sock, "default_value")
                and deserialized_val is not None
            ):
                try:
                    if type(deserialized_val) == list:
                        target_sock.default_value = deserialized_val
                    else:
                        target_sock.default_value = deserialized_val
                except Exception:
                    pass

        created_nodes[node.name] = node

    # Second-pass: resolve parent frames, set locations, retry node_tree assignments
    for n_data in data.get("nodes", []):
        node = created_nodes.get(n_data["name"])
        if node is None:
            continue

        parent_name = n_data.get("parent") or n_data.get("properties", {}).get("parent")
        if parent_name:
            parent_node = created_nodes.get(parent_name)
            if parent_node:
                try:
                    node.parent = parent_node
                except Exception as e:
                    print(f"[qna] Could not set parent on {node.name}: {e}")

        loc = n_data.get("location", [0, 0])
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            node.location = (float(loc[0]), float(loc[1]))

        # Retry node_tree assignment for GeometryNodeGroup nodes
        if n_data.get("bl_idname") != "GeometryNodeGroup":
            continue
        if node.node_tree is not None:
            continue
        nt_ref = n_data.get("properties", {}).get("node_tree")
        if nt_ref:
            resolved = _deserialize_val(nt_ref)
            if resolved is not None:
                try:
                    node.node_tree = resolved
                except Exception as e:
                    print(f"[qna] Second-pass node_tree failed for '{node.name}': {e}")

    # Links
    for l_data in data.get("links", []):
        from_node = created_nodes.get(l_data["from_node"])
        to_node = created_nodes.get(l_data["to_node"])
        if not (from_node and to_node):
            continue

        f_sock = _resolve_socket_import(
            from_node.outputs,
            l_data["from_socket"],
            l_data.get("from_socket_name", ""),
            l_data.get("from_socket_index"),
            interface_id_map,
            group_id_maps,
            from_node,
        )
        t_sock = _resolve_socket_import(
            to_node.inputs,
            l_data["to_socket"],
            l_data.get("to_socket_name", ""),
            l_data.get("to_socket_index"),
            interface_id_map,
            group_id_maps,
            to_node,
        )

        if f_sock and t_sock:
            try:
                node_tree.links.new(f_sock, t_sock)
            except Exception as e:
                print(
                    f"[qna] Failed to link {from_node.name}:{f_sock.name} "
                    f"→ {to_node.name}:{t_sock.name}: {e}"
                )

    return interface_id_map


# === JSON EXTRACTION ===


def _extract_json(text: str) -> dict | None:
    """Robustly extract a JSON object from LLM output."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    end = start
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        return json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return None


# === 2-STEP CONSTRAINED GENERATION ===


def _bg_generate_step1(base_url: str, model: str, user_prompt: str, tree_type: str):
    """
    Step 1: Ask LLM to pick nodes. Uses schema enforcement so the
    output is guaranteed to be structurally valid JSON.
    """
    try:
        _state.raw_response = "Step 1/2: Selecting nodes…\n"
        _state.dirty = True

        if tree_type == "GeometryNodeTree":
            sys_prompt = _STEP1_SYSTEM_GEOMETRY
        else:
            sys_prompt = _STEP1_SYSTEM_SHADER

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = _ollama_request_sync(
            base_url, model, messages, format_schema=_NODE_SCHEMA
        )

        if _state.abort_requested:
            _state.error = "Aborted."
            _state.gen_phase = "idle"
            _state.is_running = False
            _state.dirty = True
            return

        spec = _extract_json(raw)
        if spec is None:
            _state.error = (
                "Step 1 failed: could not parse node JSON.\n\n"
                f"Raw (first 500 chars):\n{raw[:500]}"
            )
            _state.gen_phase = "idle"
            _state.is_running = False
            _state.dirty = True
            return

        nodes = spec.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            _state.error = "Step 1 returned no nodes."
            _state.gen_phase = "idle"
            _state.is_running = False
            _state.dirty = True
            return

        _state.raw_response = f"Step 1/2: Got {len(nodes)} nodes. Building…\n"
        _state.pending_phase1 = spec
        _state.gen_phase = "building"
        _state.dirty = True

    except urllib.error.URLError as exc:
        _state.error = f"Connection error: {exc.reason}"
        _state.gen_phase = "idle"
        _state.is_running = False
        _state.dirty = True
    except Exception as exc:
        _state.error = f"Step 1 error: {exc}\n{traceback.format_exc()}"
        _state.gen_phase = "idle"
        _state.is_running = False
        _state.dirty = True


def _bg_generate_step2(
    base_url: str, model: str, user_prompt: str, socket_info: str, node_ids: list[str]
):
    """
    Step 2: Given real socket info from Blender, ask LLM to connect them.
    Node IDs are constrained via JSON schema enum.
    """
    try:
        _state.raw_response = "Step 2/2: Connecting nodes…\n"
        _state.dirty = True

        link_schema = _build_link_schema(node_ids)

        user_msg = (
            f"Material description: {user_prompt}\n\n"
            f"Here are the nodes that were created, with their EXACT socket names.\n"
            f"Use ONLY these node IDs and socket names.\n\n"
            f"{socket_info}\n\n"
            f"Specify ALL links needed to complete this material. "
            f"Make sure the final shader connects to the Surface input on 'output'."
        )

        messages = [
            {"role": "system", "content": _STEP2_SYSTEM},
            {"role": "user", "content": user_msg},
        ]

        raw = _ollama_request_sync(base_url, model, messages, format_schema=link_schema)

        if _state.abort_requested:
            _state.error = "Aborted."
            _state.gen_phase = "idle"
            _state.is_running = False
            _state.dirty = True
            return

        spec = _extract_json(raw)
        if spec is None:
            _state.error = (
                "Step 2 failed: could not parse link JSON.\n\n"
                f"Raw (first 500 chars):\n{raw[:500]}"
            )
            _state.gen_phase = "idle"
            _state.is_running = False
            _state.dirty = True
            return

        links = spec.get("links", [])
        if not isinstance(links, list):
            _state.error = "Step 2: 'links' is not a list."
            _state.gen_phase = "idle"
            _state.is_running = False
            _state.dirty = True
            return

        _state.raw_response = f"Step 2/2: Got {len(links)} links. Connecting…\n"
        _state.pending_phase2 = spec
        _state.gen_phase = "linking"
        _state.dirty = True

    except urllib.error.URLError as exc:
        _state.error = f"Connection error: {exc.reason}"
        _state.gen_phase = "idle"
        _state.is_running = False
        _state.dirty = True
    except Exception as exc:
        _state.error = f"Step 2 error: {exc}\n{traceback.format_exc()}"
        _state.gen_phase = "idle"
        _state.is_running = False
        _state.dirty = True


# === SOCKET INFO READER ===


def _read_socket_info(
    node_tree: bpy.types.NodeTree, id_to_node: dict[str, bpy.types.Node]
) -> str:
    """
    Read REAL socket names from built Blender nodes and format them
    as a clear reference for the LLM's step 2.
    """
    lines = []
    for nid, node in id_to_node.items():
        label_part = f' "{node.label}"' if node.label else ""
        lines.append(f"[{nid}] {node.bl_idname}{label_part}")

        outs = [s.name for s in node.outputs if s.enabled and not s.hide]
        if outs:
            lines.append(f"  OUTPUTS: {', '.join(outs)}")
        else:
            lines.append(f"  OUTPUTS: (none)")

        ins = [s.name for s in node.inputs if s.enabled and not s.hide]
        if ins:
            lines.append(f"  INPUTS: {', '.join(ins)}")
        else:
            lines.append(f"  INPUTS: (none)")
        lines.append("")

    return "\n".join(lines)


# === NODE BUILDER (type-safe) ===

_SOCKET_ALIASES = {
    "fac": "Factor",
    "fac.": "Factor",
    "shader_1": "Shader",
    "shader_2": "Shader",
    "shader 1": "Shader",
    "shader 2": "Shader",
    "base color": "Base Color",
    "val": "Value",
    "col": "Color",
    "vec": "Vector",
    "bsdf": "BSDF",
    "result": "Result",
    "dist": "Distance",
    "rough": "Roughness",
    "emission": "Emission",
    "geo": "Geometry",
    "geometry": "Geometry",
    "pts": "Points",
    "points": "Points",
    "inst": "Instances",
    "instances": "Instances",
    "mesh": "Mesh",
    "position": "Position",
    "pos": "Position",
    "normal": "Normal",
    "norm": "Normal",
    "rotation": "Rotation",
    "rot": "Rotation",
    "scale": "Scale",
    "density": "Density",
    "selection": "Selection",
    "sel": "Selection",
    "attribute": "Attribute",
    "attr": "Attribute",
    "name": "Name",
    "material": "Material",
    "mat": "Material",
    "object": "Object",
    "obj": "Object",
    "collection": "Collection",
    "coll": "Collection",
    "image": "Image",
    "img": "Image",
    "transform": "Transform",
    "matrix": "Matrix",
    "invert": "Invert",
    "multiply": "Multiply",
    "mul": "Multiply",
    "add": "Add",
    "subtract": "Subtract",
    "sub": "Subtract",
    "divide": "Divide",
    "div": "Divide",
    "power": "Power",
    "pow": "Power",
    "absolute": "Absolute",
    "abs": "Absolute",
    "min": "Min",
    "max": "Max",
    "clamp": "Clamp",
    "mix": "Mix",
    "blend": "Mix",
    "compare": "Compare",
    "cmp": "Compare",
    "random": "Random",
    "rand": "Random",
    "noise": "Noise",
    "curve": "Curve",
    "spline": "Spline",
    "handle": "Handle Type",
    "resolution": "Resolution",
    "trim": "Trim",
    "start": "Start",
    "end": "End",
    "length": "Length",
    "radius": "Radius",
    "tangent": "Tangent",
    "curvature": "Curvature",
    "twist": "Twist",
    "tilt": "Tilt",
    "weight": "Weight",
    "id": "ID",
    "boolean": "Boolean",
    "bool": "Boolean",
    "int": "Integer",
    "integer": "Integer",
    "float": "Float",
    "string": "String",
    "color": "Color",
    "vector": "Vector",
}


def _find_socket(sockets, name: str):
    """Find socket with multi-level fallback."""
    if not name:
        return None
    for s in sockets:
        if s.name == name:
            return s
    name_lower = name.lower()
    for s in sockets:
        if s.name.lower() == name_lower:
            return s
    alias = _SOCKET_ALIASES.get(name_lower)
    if alias:
        for s in sockets:
            if s.name == alias:
                return s
    name_stripped = name_lower.replace("_", " ").replace("-", " ").strip()
    for s in sockets:
        if s.name.lower().replace("_", " ").replace("-", " ") == name_stripped:
            return s
    for s in sockets:
        sn = s.name.lower()
        if name_lower in sn or sn in name_lower:
            return s
    return None


def _set_socket_value(socket, value):
    if not hasattr(socket, "default_value"):
        return
    try:
        if isinstance(value, (list, tuple)):
            dv = socket.default_value
            if hasattr(dv, "__len__"):
                for i in range(min(len(value), len(dv))):
                    dv[i] = value[i]
            else:
                socket.default_value = value[0]
        elif isinstance(value, (int, float)):
            socket.default_value = value
        elif isinstance(value, str):
            socket.default_value = value
    except Exception:
        pass


def _safe_dict(val, label: str = "") -> dict:
    """Coerce a value to a dict. Handles list-of-pairs, list-of-dicts, etc."""
    if isinstance(val, dict):
        return val
    if isinstance(val, list):
        result = {}
        for item in val:
            if isinstance(item, dict) and "name" in item:
                result[item["name"]] = item.get("value", item.get("default_value"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                result[str(item[0])] = item[1]
        return result
    return {}


def _ensure_geometry_structure(tree, id_to_node: dict, log: list) -> dict:
    """
    For geometry node trees: guarantee group_input, group_output, and join_geo
    exist as Blender nodes, creating any that are missing.
    Returns the updated id_to_node dict.
    """

    def _find_or_create(bl_id, node_id, label):
        node = id_to_node.get(node_id)
        if node:
            return node
        for n in tree.nodes:
            if n.bl_idname == bl_id:
                n.name = node_id
                id_to_node[node_id] = n
                return n
        node = tree.nodes.new(bl_id)
        node.name = node_id
        node.label = label
        id_to_node[node_id] = node
        log.append(f"AUTO  Created {bl_id} '{node_id}'")
        return node

    _find_or_create("NodeGroupInput", "group_input", "Group Input")
    _find_or_create("NodeGroupOutput", "group_output", "Group Output")
    # Only create join_geo if it doesn't already exist — some setups don't need it
    if "join_geo" not in id_to_node:
        for n in tree.nodes:
            if n.bl_idname == "GeometryNodeJoinGeometry":
                id_to_node["join_geo"] = n
                break
        else:
            _find_or_create("GeometryNodeJoinGeometry", "join_geo", "Join Geometry")
    return id_to_node


def _fix_geometry_output_links(tree, id_to_node: dict, log: list) -> None:
    """
    After the LLM has created links for a geometry node tree, enforce:
      1. group_input.Geometry feeds into the geometry pipeline
      2. group_output.Geometry receives exactly one Geometry link
      3. join_geo acts as the final collector if it exists

    Only intervenes when group_output has NO geometry input (missing link)
    or has multiple conflicting inputs. Does NOT reroute valid topologies.
    """
    group_input = id_to_node.get("group_input")
    group_output = id_to_node.get("group_output")
    join_geo = id_to_node.get("join_geo")

    if not all([group_input, group_output]):
        log.append(
            "WARN  geometry structure incomplete — skipping output link enforcement"
        )
        return

    geo_out_input = _find_socket(group_output.inputs, "Geometry")
    if not geo_out_input:
        log.append("WARN  could not find Geometry output socket — skipping")
        return

    # Check what's currently connected to group_output.Geometry
    existing_links = [lk for lk in tree.links if lk.to_socket == geo_out_input]

    # If nothing is connected, try to wire up join_geo or group_input
    if not existing_links:
        if join_geo:
            join_geo_out = _find_socket(join_geo.outputs, "Geometry")
            join_geo_in = _find_socket(join_geo.inputs, "Geometry")
            gi_geo_out = _find_socket(group_input.outputs, "Geometry")
            if join_geo_out and join_geo_in:
                tree.links.new(join_geo_out, geo_out_input)
                log.append("AUTO  Connected join_geo.Geometry → group_output")
                if gi_geo_out:
                    already = any(
                        lk.from_socket == gi_geo_out and lk.to_node == join_geo
                        for lk in tree.links
                    )
                    if not already:
                        tree.links.new(gi_geo_out, join_geo_in)
                        log.append("AUTO  Connected group_input.Geometry → join_geo")
        else:
            gi_geo_out = _find_socket(group_input.outputs, "Geometry")
            if gi_geo_out:
                tree.links.new(gi_geo_out, geo_out_input)
                log.append(
                    "AUTO  Connected group_input.Geometry → group_output (no join_geo)"
                )

    # If multiple things feed into group_output.Geometry, keep only the last one
    elif len(existing_links) > 1:
        for lk in existing_links[:-1]:
            tree.links.remove(lk)
        log.append("AUTO  Removed duplicate links to group_output.Geometry")


def build_nodes_only(
    node_tree: bpy.types.NodeTree, spec: dict, clear_existing: bool = True
) -> tuple[dict, list[str]]:
    """
    Build nodes from spec WITHOUT creating links.
    Returns (id_to_node dict, log list).
    Supports dynamic node items (capture_items, store_items, repeat_items, simulation_items)
    for geometry nodes like Store Named Attribute, Capture Attribute, Repeat Zones, etc.
    """
    log: list[str] = []

    if clear_existing:
        node_tree.nodes.clear()
        log.append("Cleared existing nodes.")

    id_to_node: dict[str, bpy.types.Node] = {}
    nodes_data = spec.get("nodes", [])
    if not isinstance(nodes_data, list):
        log.append(
            f"WARN  'nodes' is {type(nodes_data).__name__}, expected list. Skipping."
        )
        return id_to_node, log

    for i, ns in enumerate(nodes_data):
        if not isinstance(ns, dict):
            log.append(f"WARN  Node entry {i} is {type(ns).__name__}, skipping.")
            continue

        nid = str(ns.get("id", f"node_{i}"))
        bl_idname = str(ns.get("bl_idname", ""))

        try:
            node = node_tree.nodes.new(type=bl_idname)
        except Exception as exc:
            log.append(f"FAIL  {bl_idname}: {exc}")
            continue

        node.name = nid
        label = ns.get("label", "")
        if label:
            node.label = str(label)
        log.append(f"OK  {bl_idname} → '{nid}'")

        # Properties
        props = _safe_dict(ns.get("properties", {}), f"{nid}.properties")
        for prop_name, prop_val in props.items():
            if hasattr(node, prop_name):
                try:
                    setattr(node, prop_name, prop_val)
                except Exception as exc:
                    log.append(f"  WARN  {nid}.{prop_name}={prop_val}: {exc}")

        # Dynamic node items (Capture Attribute, Store Named Attribute, Repeat Zones, Simulation Zones)
        # MUST be done before socket resolution so that dynamic sockets appear
        for col_name in (
            "capture_items",
            "store_items",
            "repeat_items",
            "simulation_items",
        ):
            if col_name in ns and hasattr(node, col_name):
                col = getattr(node, col_name)
                items_data = ns[col_name]
                if isinstance(items_data, list):
                    col.clear()
                    for item_data in items_data:
                        if not isinstance(item_data, dict):
                            continue
                        dt = item_data.get("data_type", "FLOAT")
                        name = item_data.get("name", "")
                        try:
                            new_item = col.new(dt, name)
                            for k, v in item_data.items():
                                if hasattr(new_item, k):
                                    d_v = _deserialize_val(v)
                                    if d_v is not None:
                                        try:
                                            setattr(new_item, k, d_v)
                                        except Exception:
                                            pass
                        except Exception as exc:
                            log.append(
                                f"  WARN  Failed to create {col_name} item on {nid}: {exc}"
                            )

        # Input defaults
        inputs = _safe_dict(ns.get("inputs", {}), f"{nid}.inputs")
        for sock_name, sock_val in inputs.items():
            sock = _find_socket(node.inputs, sock_name)
            if sock:
                _set_socket_value(sock, sock_val)
            else:
                log.append(f"  WARN  Input '{sock_name}' not found on {nid}")

        id_to_node[nid] = node

    # Auto-layout
    _auto_layout_from_ids(list(id_to_node.keys()), id_to_node)

    return id_to_node, log


def create_links(
    node_tree: bpy.types.NodeTree, spec: dict, id_to_node: dict
) -> list[str]:
    """Create links from a links spec, with smart resolution."""
    log: list[str] = []
    links_data = spec.get("links", [])
    if not isinstance(links_data, list):
        log.append(f"WARN  'links' is {type(links_data).__name__}, expected list.")
        return log

    link_ok = 0
    link_fail = 0

    for i, ls in enumerate(links_data):
        if not isinstance(ls, dict):
            log.append(f"WARN  Link entry {i} is not a dict, skipping.")
            link_fail += 1
            continue

        fn_id = str(ls.get("from_node", ""))
        fs_name = str(ls.get("from_socket", ""))
        tn_id = str(ls.get("to_node", ""))
        ts_name = str(ls.get("to_socket", ""))

        # Dot-notation fix
        fn_id, fs_name = _split_combined_ref(fn_id, fs_name, id_to_node)
        tn_id, ts_name = _split_combined_ref(tn_id, ts_name, id_to_node)

        fn = id_to_node.get(fn_id)
        tn = id_to_node.get(tn_id)

        if not fn:
            fn, fn_id = _find_node_by_label(fn_id, id_to_node)
        if not tn:
            tn, tn_id = _find_node_by_label(tn_id, id_to_node)

        if not fn:
            log.append(f"FAIL  from_node '{ls.get('from_node', '')}' not found")
            link_fail += 1
            continue
        if not tn:
            log.append(f"FAIL  to_node '{ls.get('to_node', '')}' not found")
            link_fail += 1
            continue

        # Socket resolution
        fs = _resolve_output(fn, fs_name, tn, ts_name)
        ts = _resolve_input(tn, ts_name, fn, fs_name)

        if not fs:
            avail = [s.name for s in fn.outputs if s.enabled]
            log.append(f"FAIL  output '{fs_name}' not on '{fn_id}' (has: {avail})")
            link_fail += 1
            continue
        if not ts:
            avail = [s.name for s in tn.inputs if s.enabled]
            log.append(f"FAIL  input '{ts_name}' not on '{tn_id}' (has: {avail})")
            link_fail += 1
            continue

        try:
            node_tree.links.new(fs, ts)
            link_ok += 1
            # Log auto-corrections
            orig_fs = ls.get("from_socket", "")
            orig_ts = ls.get("to_socket", "")
            if fs.name != orig_fs or ts.name != orig_ts:
                log.append(f"  AUTO  {fn_id}.{fs.name} → {tn_id}.{ts.name}")
        except Exception as exc:
            log.append(f"FAIL  {fn_id}→{tn_id}: {exc}")
            link_fail += 1

    log.append(f"Links: {link_ok} connected, {link_fail} failed.")
    return log


# === LINK RESOLUTION HELPERS ===


def _split_combined_ref(
    node_ref: str, socket_name: str, id_to_node: dict
) -> tuple[str, str]:
    if not node_ref or node_ref in id_to_node:
        return node_ref, socket_name
    for sep in [".", "/", " -> ", "->", " → ", "→", ":"]:
        if sep in node_ref:
            parts = node_ref.split(sep, 1)
            candidate = parts[0].strip()
            sock = parts[1].strip()
            if candidate in id_to_node:
                return candidate, sock if not socket_name else socket_name
            for nid, node in id_to_node.items():
                if nid.lower() == candidate.lower():
                    return nid, sock if not socket_name else socket_name
                if node.label and node.label.lower() == candidate.lower():
                    return nid, sock if not socket_name else socket_name
    # Case-insensitive id match
    for nid in id_to_node:
        if nid.lower() == node_ref.lower():
            return nid, socket_name
    return node_ref, socket_name


def _find_node_by_label(ref: str, id_to_node: dict):
    if not ref:
        return None, ref
    ref_lower = ref.lower()
    for nid, node in id_to_node.items():
        if nid.lower() == ref_lower:
            return node, nid
    for nid, node in id_to_node.items():
        if node.label and node.label.lower() == ref_lower:
            return node, nid
    for nid, node in id_to_node.items():
        label = (node.label or "").lower()
        if label and (ref_lower in label or label in ref_lower):
            return node, nid
        if ref_lower in nid.lower() or nid.lower() in ref_lower:
            return node, nid
    return None, ref


_DEFAULT_OUTPUTS = {
    "ShaderNodeBsdfPrincipled": "BSDF",
    "ShaderNodeBsdfDiffuse": "BSDF",
    "ShaderNodeBsdfGlossy": "BSDF",
    "ShaderNodeBsdfGlass": "BSDF",
    "ShaderNodeBsdfTransparent": "BSDF",
    "ShaderNodeBsdfTranslucent": "BSDF",
    "ShaderNodeBsdfAnisotropic": "BSDF",
    "ShaderNodeBsdfToon": "BSDF",
    "ShaderNodeBsdfSheen": "BSDF",
    "ShaderNodeSubsurfaceScattering": "BSDF",
    "ShaderNodeEmission": "Emission",
    "ShaderNodeBackground": "Background",
    "ShaderNodeMixShader": "Shader",
    "ShaderNodeAddShader": "Shader",
    "ShaderNodeTexNoise": "Factor",
    "ShaderNodeTexVoronoi": "Distance",
    "ShaderNodeTexWave": "Factor",
    "ShaderNodeTexGradient": "Factor",
    "ShaderNodeTexChecker": "Color",
    "ShaderNodeTexBrick": "Color",
    "ShaderNodeTexMagic": "Color",
    "ShaderNodeTexImage": "Color",
    "ShaderNodeTexWhiteNoise": "Value",
    "ShaderNodeFresnel": "Factor",
    "ShaderNodeLayerWeight": "Fresnel",
    "ShaderNodeBump": "Normal",
    "ShaderNodeNormalMap": "Normal",
    "ShaderNodeDisplacement": "Displacement",
    "ShaderNodeMapping": "Vector",
    "ShaderNodeTexCoord": "UV",
    "ShaderNodeMath": "Value",
    "ShaderNodeVectorMath": "Vector",
    "ShaderNodeMapRange": "Result",
    "ShaderNodeClamp": "Result",
    "ShaderNodeMix": "Result",
    "ShaderNodeInvert": "Color",
    "ShaderNodeHueSaturation": "Color",
    "ShaderNodeBrightContrast": "Color",
    "ShaderNodeGamma": "Color",
    "ShaderNodeValToRGB": "Color",
    "ShaderNodeRGBToBW": "Val",
    "ShaderNodeCombineXYZ": "Vector",
    "ShaderNodeSeparateXYZ": "X",
    "ShaderNodeRGB": "Color",
    "ShaderNodeValue": "Value",
    "ShaderNodeGeometry": "Position",
    "ShaderNodeAmbientOcclusion": "AO",
    "ShaderNodeWireframe": "Factor",
    "ShaderNodeShaderToRGB": "Color",
    "NodeGroupInput": "Geometry",
    "GeometryNodeMeshCube": "Mesh",
    "GeometryNodeMeshGrid": "Mesh",
    "GeometryNodeMeshUVSphere": "Mesh",
    "GeometryNodeMeshIcoSphere": "Mesh",
    "GeometryNodeMeshCylinder": "Mesh",
    "GeometryNodeMeshCone": "Mesh",
    "GeometryNodeTransform": "Geometry",
    "GeometryNodeSetPosition": "Geometry",
    "GeometryNodeSetMaterial": "Geometry",
    "GeometryNodeJoinGeometry": "Geometry",
    "GeometryNodeInstanceOnPoints": "Instances",
    "GeometryNodeRealizeInstances": "Geometry",
    "GeometryNodeDistributePointsOnFaces": "Points",
    "GeometryNodeInputPosition": "Position",
    "GeometryNodeInputNormal": "Normal",
    "GeometryNodeInputIndex": "Index",
    "GeometryNodeInputNamedAttribute": "Value",
    "GeometryNodeInputSceneTime": "Time",
    "GeometryNodeInputActiveCamera": "Camera",
    "GeometryNodeObjectInfo": "Geometry",
    "GeometryNodeCollectionInfo": "Instances",
    "GeometryNodeCameraInfo": "View Vector",
    "GeometryNodeBoolean": "Mesh",
    "GeometryNodeExtrudeMesh": "Mesh",
    "GeometryNodeDeleteGeometry": "Geometry",
    "GeometryNodeSplitToInstances": "Instances",
    "GeometryNodeSeparateGeometry": "Geometry",
    "GeometryNodeTriangulate": "Mesh",
    "GeometryNodeConvexHull": "Geometry",
    "GeometryNodeBoundingBox": "Bounding Box",
    "GeometryNodeCurveToMesh": "Mesh",
    "GeometryNodeCurveToPoints": "Points",
    "GeometryNodePointsToCurves": "Curves",
    "GeometryNodeResampleCurve": "Curve",
    "GeometryNodeSubdivideCurve": "Curve",
    "GeometryNodeTrimCurve": "Curve",
    "GeometryNodeFilletCurve": "Curve",
    "GeometryNodeInterpolateCurves": "Curve",
    "GeometryNodeOffsetPointInCurve": "Vector",
    "GeometryNodeSplineParameter": "Position",
    "GeometryNodeSetCurveRadius": "Curve",
    "GeometryNodeSetCurveTilt": "Curve",
    "GeometryNodeSetHandleType": "Curve",
    "GeometryNodeSetSplineResolution": "Curve",
    "GeometryNodeReverseCurve": "Curve",
    "GeometryNodeStoreNamedAttribute": "Geometry",
    "GeometryNodeCaptureAttribute": "Geometry",
    "GeometryNodeNamedAttribute": "Value",
    "GeometryNodeViewer": "Geometry",
    "GeometryNodeSimulationInput": "Geometry",
    "GeometryNodeSimulationOutput": "Geometry",
    "GeometryNodeRepeatInput": "Geometry",
    "GeometryNodeRepeatOutput": "Geometry",
    "GeometryNodeForEachElementInput": "Geometry",
    "GeometryNodeForEachElementOutput": "Geometry",
    "FunctionNodeTransformPoint": "Point",
    "FunctionNodeTransformDirection": "Direction",
    "FunctionNodeInvertMatrix": "Inverse",
    "FunctionNodeCombineMatrix": "Matrix",
    "FunctionNodeSeparateMatrix": "Row 1",
    "FunctionNodeRotateEuler": "Rotation",
    "FunctionNodeAlignEulerToVector": "Rotation",
    "FunctionNodeProjectPoint": "Result",
    "FunctionNodeRandomValue": "Value",
    "FunctionNodeCompare": "Result",
    "GeometryNodeMergeByDistance": "Geometry",
    "GeometryNodeDualMesh": "Mesh",
    "GeometryNodeCurvePrimitiveLine": "Curve",
    "GeometryNodeCurvePrimitiveCircle": "Curve",
    "GeometryNodeCurvePrimitiveRectangle": "Curve",
    "GeometryNodeCurvePrimitiveQuadrilateral": "Curve",
    "GeometryNodeMeshLine": "Mesh",
    "GeometryNodeMeshCircle": "Mesh",
    "GeometryNodeMeshTriangle": "Mesh",
    "GeometryNodeImageTexture": "Color",
    "GeometryNodeFieldAtIndex": "Value",
    "GeometryNodeDomainSize": "Size",
    "GeometryNodeRemoveAttribute": "Geometry",
    "GeometryNodeSplineType": "Curve",
    "GeometryNodeIsViewport": "Is Viewport",
    "GeometryNodeInstancesToPoints": "Points",
    "GeometryNodePointsToVertices": "Mesh",
    "GeometryNodeImageInfo": "Width",
}

_DEFAULT_INPUTS = {
    "ShaderNodeOutputMaterial": "Surface",
    "ShaderNodeMixShader": "Factor",
    "ShaderNodeBsdfPrincipled": "Base Color",
    "ShaderNodeBump": "Height",
    "ShaderNodeDisplacement": "Height",
    "ShaderNodeMapping": "Vector",
    "ShaderNodeMath": "Value",
    "ShaderNodeMix": "Factor",
    "ShaderNodeMapRange": "Value",
    "ShaderNodeClamp": "Value",
    "ShaderNodeValToRGB": "Factor",
    "ShaderNodeRGBToBW": "Color",
    "NodeGroupOutput": "Geometry",
    "GeometryNodeSetPosition": "Geometry",
    "GeometryNodeTransform": "Geometry",
    "GeometryNodeJoinGeometry": "Geometry",
    "GeometryNodeInstanceOnPoints": "Points",
}


def _resolve_output(from_node, from_socket_name, to_node=None, to_socket_name=""):
    if from_socket_name:
        s = _find_socket(from_node.outputs, from_socket_name)
        if s:
            return s
    default = _DEFAULT_OUTPUTS.get(from_node.bl_idname, "")
    if default:
        for s in from_node.outputs:
            if s.name == default and s.enabled:
                return s
    if to_node and to_socket_name:
        target = _find_socket(to_node.inputs, to_socket_name)
        if target:
            for s in from_node.outputs:
                if s.enabled and s.bl_idname == target.bl_idname:
                    return s
    for s in from_node.outputs:
        if s.enabled:
            return s
    return None


def _resolve_input(to_node, to_socket_name, from_node=None, from_socket_name=""):
    if to_socket_name:
        s = _find_socket(to_node.inputs, to_socket_name)
        if s:
            return s
    if from_node:
        src = None
        if from_socket_name:
            src = _find_socket(from_node.outputs, from_socket_name)
        if not src:
            default = _DEFAULT_OUTPUTS.get(from_node.bl_idname, "")
            if default:
                for s in from_node.outputs:
                    if s.name == default and s.enabled:
                        src = s
                        break
        if not src and from_node.outputs:
            src = from_node.outputs[0]
        if src:
            for s in to_node.inputs:
                if s.enabled and s.bl_idname == src.bl_idname and not s.is_linked:
                    return s
            for s in to_node.inputs:
                if s.enabled and s.bl_idname == src.bl_idname:
                    return s
    default = _DEFAULT_INPUTS.get(to_node.bl_idname, "")
    if default:
        for s in to_node.inputs:
            if s.name == default and s.enabled:
                return s
    for s in to_node.inputs:
        if s.enabled and not s.is_linked:
            return s
    return None


# === AUTO-LAYOUT ===


def _auto_layout_from_ids(
    ids: list[str], id_to_node: dict, links_spec: list | None = None
):
    """Position nodes left-to-right. If no links provided, simple grid."""
    if not ids:
        return

    if links_spec:
        dependents: dict[str, set[str]] = {nid: set() for nid in ids}
        dependencies: dict[str, set[str]] = {nid: set() for nid in ids}
        for ls in links_spec:
            if not isinstance(ls, dict):
                continue
            fn = str(ls.get("from_node", ""))
            tn = str(ls.get("to_node", ""))
            if fn in dependents and tn in dependencies:
                dependents[fn].add(tn)
                dependencies[tn].add(fn)

        in_degree = {nid: len(deps) for nid, deps in dependencies.items()}
        queue = [nid for nid, d in in_degree.items() if d == 0]
        columns: list[list[str]] = []
        visited = set()
        while queue:
            columns.append(list(queue))
            visited.update(queue)
            next_q = []
            for nid in queue:
                for dep in dependents.get(nid, []):
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0 and dep not in visited:
                        next_q.append(dep)
            queue = next_q
        remaining = [nid for nid in ids if nid not in visited]
        if remaining:
            columns.append(remaining)
    else:
        # Simple grid: 3 nodes per column
        columns = []
        for i in range(0, len(ids), 3):
            columns.append(ids[i : i + 3])

    x_spacing = 300
    y_spacing = 200
    for col_i, col in enumerate(columns):
        x = col_i * x_spacing
        col_height = len(col) * y_spacing
        for row_i, nid in enumerate(col):
            node = id_to_node.get(nid)
            if node:
                node.location.x = x
                node.location.y = (col_height / 2) - row_i * y_spacing


# === POLL TIMER (multi-phase) ===


def _poll_timer() -> float | None:
    """100ms poller: handles display updates and multi-phase generation."""

    # --- Phase: building (nodes created, need to read sockets & start step 2) ---
    if _state.gen_phase == "building" and _state.pending_phase1 is not None:
        spec = _state.pending_phase1
        _state.pending_phase1 = None

        tree = _find_active_tree()
        if tree is None:
            _state.error = "No active node tree found."
            _state.gen_phase = "idle"
            _state.is_running = False
            _state.dirty = True
        else:
            clear = _state.gen_clear
            try:
                id_to_node, log = build_nodes_only(tree, spec, clear_existing=clear)
                if tree.bl_idname == "GeometryNodeTree":
                    id_to_node = _ensure_geometry_structure(tree, id_to_node, log)
                _state.build_log = log

                if not id_to_node:
                    _state.error = "No nodes were created.\n" + "\n".join(log)
                    _state.gen_phase = "idle"
                    _state.is_running = False
                    _state.dirty = True
                    return 0.1

                # Read REAL socket names from Blender
                socket_info = _read_socket_info(tree, id_to_node)
                node_ids = list(id_to_node.keys())

                # Save debug info
                _save_debug_text("QNA_Step1_Nodes.json", json.dumps(spec, indent=2))
                _save_debug_text("QNA_Step2_SocketInfo.txt", socket_info)

                desc = spec.get("description", "")
                _state.raw_response = (
                    f"Step 1 complete: {len(id_to_node)} nodes built.\n"
                )
                if desc:
                    _state.raw_response += f"{desc}\n"
                _state.raw_response += (
                    "\n".join(log) + "\n\nStep 2/2: Asking LLM to connect them…\n"
                )

                # Store context for step 2
                _state.gen_node_ids = node_ids
                _state.gen_socket_info = socket_info
                _state.gen_phase = "connecting"

                # Launch step 2 in background
                thread = threading.Thread(
                    target=_bg_generate_step2,
                    args=(
                        _state.gen_base_url,
                        _state.gen_model,
                        _state.gen_user_prompt,
                        socket_info,
                        node_ids,
                    ),
                    daemon=True,
                )
                thread.start()

            except Exception as exc:
                _state.error = f"Build failed: {exc}\n{traceback.format_exc()}"
                _state.gen_phase = "idle"
                _state.is_running = False

            _state.dirty = True

    # --- Phase: linking (links spec ready, create them) ---
    if _state.gen_phase == "linking" and _state.pending_phase2 is not None:
        spec = _state.pending_phase2
        _state.pending_phase2 = None

        tree = _find_active_tree()
        if tree is None:
            _state.error = "No active node tree found for linking."
        else:
            try:
                # Rebuild id_to_node from current tree
                id_to_node = {node.name: node for node in tree.nodes}

                _save_debug_text("QNA_Step2_Links.json", json.dumps(spec, indent=2))

                link_log = create_links(tree, spec, id_to_node)

                if tree.bl_idname == "GeometryNodeTree":
                    _fix_geometry_output_links(tree, id_to_node, link_log)

                # Re-layout now that we have links
                links_data = spec.get("links", [])
                _auto_layout_from_ids(list(id_to_node.keys()), id_to_node, links_data)

                # Combine logs
                prev = _state.raw_response
                _state.raw_response = (
                    prev
                    + "\n--- Link Log ---\n"
                    + "\n".join(link_log)
                    + "\n\nDone! Debug saved to text blocks."
                )
                _state.build_log.extend(link_log)

            except Exception as exc:
                _state.error = f"Linking failed: {exc}\n{traceback.format_exc()}"

        _state.gen_phase = "idle"
        _state.is_running = False
        _state.dirty = True

    # --- Display update ---
    if _state.dirty:
        _state.dirty = False
        wrap = _pref("response_wrap_width", 62)
        wrapped: list[str] = []
        for paragraph in _state.raw_response.split("\n"):
            if paragraph.strip() == "":
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(paragraph, width=wrap) or [""])
        _state.response_lines = wrapped

        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "NODE_EDITOR":
                    area.tag_redraw()

    # Unregister when fully done
    if not _state.is_running and _state.gen_phase == "idle":
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "NODE_EDITOR":
                    area.tag_redraw()
        return None

    return 0.1


def _find_active_tree() -> bpy.types.NodeTree | None:
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "NODE_EDITOR":
                for space in area.spaces:
                    if space.type == "NODE_EDITOR" and space.edit_tree:
                        return space.edit_tree
    return None


def _save_debug_text(name: str, content: str):
    if name in bpy.data.texts:
        txt = bpy.data.texts[name]
        txt.clear()
    else:
        txt = bpy.data.texts.new(name)
    txt.write(content)


def _restore_tree_from_snapshot(tree, snapshot: dict) -> list[str]:
    """
    Restore a node tree from a serialize_node_tree() snapshot.
    Clears the tree then recreates all nodes and links by their original names.
    """
    log: list[str] = []
    tree.nodes.clear()

    name_to_node: dict = {}
    for nd in snapshot.get("nodes", []):
        bl_id = nd.get("bl_idname", "")
        if not bl_id:
            continue
        try:
            node = tree.nodes.new(bl_id)
            node.name = nd.get("name", "")
            node.label = nd.get("label", "")
            loc = nd.get("location", [0.0, 0.0])
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                node.location = (float(loc[0]), float(loc[1]))
            name_to_node[node.name] = node
        except Exception as exc:
            log.append(f"WARN  Restore node '{nd.get('name', '')}' failed: {exc}")

    for lk in snapshot.get("links", []):
        fn = lk.get("from_node", "")
        fs = lk.get("from_socket", "")
        tn = lk.get("to_node", "")
        ts = lk.get("to_socket", "")
        from_node = name_to_node.get(fn)
        to_node = name_to_node.get(tn)
        if not from_node or not to_node:
            continue
        from_sock = _find_socket(from_node.outputs, fs)
        to_sock = _find_socket(to_node.inputs, ts)
        if from_sock and to_sock:
            tree.links.new(from_sock, to_sock)

    log.append(f"Reverted to snapshot: {len(name_to_node)} nodes restored.")
    return log


# === OPERATORS ===


class QNA_OT_RefreshModels(bpy.types.Operator):
    """Fetch available models from Ollama, filter embedding models, and auto-select the best ones"""

    bl_idname = "qna.refresh_models"
    bl_label = "Refresh Models"
    bl_options = {"REGISTER"}

    def execute(self, context):
        base_url = _pref("ollama_base_url", "http://localhost:11434")
        try:
            url = f"{base_url.rstrip('/')}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            all_names = sorted(m["name"] for m in data.get("models", []))
            usable = [n for n in all_names if not _is_embed_model(n)]
            filtered = len(all_names) - len(usable)

            if not usable:
                self.report(
                    {"WARNING"}, "No usable models found (all were embedding models)"
                )
                return {"CANCELLED"}

            _rebuild_model_lists(all_names)

            # Auto-select best model for each role, but don't override a valid
            # manual choice the user already made
            scene = context.scene
            best_gen = _best_model_for(usable, "gen")
            best_ana = _best_model_for(usable, "ana")

            gen_list = [item[0] for item in _model_items_generate if item[0] != "none"]
            ana_list = [item[0] for item in _model_items_analyze if item[0] != "none"]

            if scene.qna_model_generate not in gen_list and best_gen:
                scene.qna_model_generate = best_gen
            if scene.qna_model_analyze not in ana_list and best_ana:
                scene.qna_model_analyze = best_ana

            msg = f"{len(usable)} models loaded"
            if filtered:
                msg += f" ({filtered} embedding model(s) hidden)"
            if best_gen:
                msg += f" · best generate: {best_gen.split(':')[0]}"
            self.report({"INFO"}, msg)

        except urllib.error.URLError:
            self.report({"ERROR"}, "Could not reach Ollama — is it running?")
        except Exception as exc:
            self.report({"ERROR"}, f"Refresh failed: {exc}")
        return {"FINISHED"}


class QNA_OT_Abort(bpy.types.Operator):
    """Stop the current generation or analysis"""

    bl_idname = "qna.abort"
    bl_label = "Abort"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _state.is_running

    def execute(self, context):
        _state.abort_requested = True
        return {"FINISHED"}


class QNA_OT_Revert(bpy.types.Operator):
    """Restore the node tree to its state before the last generation"""

    bl_idname = "qna.revert"
    bl_label = "Undo Last Generation"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return _state.pre_gen_snapshot is not None and not _state.is_running

    def execute(self, context):
        if _state.pre_gen_snapshot is None:
            self.report({"WARNING"}, "No snapshot available.")
            return {"CANCELLED"}

        tree = _find_active_tree()
        if tree is None:
            self.report({"ERROR"}, "No active node tree found.")
            return {"CANCELLED"}

        snapshot = _state.pre_gen_snapshot
        _state.pre_gen_snapshot = None  # consume immediately so button hides

        log = _restore_tree_from_snapshot(tree, snapshot)
        _state.raw_response = "\n".join(log)
        _state.response_lines = log
        _state.dirty = True

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "NODE_EDITOR":
                    area.tag_redraw()

        self.report({"INFO"}, "Node tree reverted to pre-generation state.")
        return {"FINISHED"}


class QNA_OT_AskQwen(bpy.types.Operator):
    """Analyze the current node tree"""

    bl_idname = "qna.ask_qwen"
    bl_label = "Ask Qwen"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        if not space or space.type != "NODE_EDITOR":
            return False
        if _state.is_running:
            return False
        return space.edit_tree is not None

    def execute(self, context):
        tree = context.space_data.edit_tree
        question = context.scene.qna_user_question.strip()
        if not question:
            _state.error = "Please enter a question first."
            _state.response_lines = []
            _state.raw_response = ""
            return {"CANCELLED"}

        _state.response_lines = []
        _state.raw_response = ""
        _state.error = ""
        _state.build_log = []
        _state.abort_requested = False
        _state.is_running = True
        _state.dirty = False

        try:
            tree_json = json.dumps(serialize_node_tree(tree), indent=2)
        except Exception as exc:
            _state.error = f"Serialization failed: {exc}"
            _state.is_running = False
            return {"CANCELLED"}

        base_url = _pref("ollama_base_url", "http://localhost:11434")
        model = context.scene.qna_model_analyze or _pref(
            "ollama_model_analyze", "qwen2.5vl:7b"
        )
        _state.loaded_base_url = base_url
        _state.loaded_model = model
        _state.model_is_loaded = True
        messages = [
            {"role": "system", "content": _ANALYZE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Node tree JSON:\n```json\n{tree_json}\n```\n\nQuestion: {question}"
                ),
            },
        ]

        thread = threading.Thread(
            target=_bg_analyze,
            args=(base_url, model, messages),
            daemon=True,
        )
        thread.start()
        if not bpy.app.timers.is_registered(_poll_timer):
            bpy.app.timers.register(_poll_timer, first_interval=0.1)
        return {"FINISHED"}


class QNA_OT_GenerateNodes(bpy.types.Operator):
    """Generate nodes using 2-step constrained pipeline"""

    bl_idname = "qna.generate_nodes"
    bl_label = "Generate Nodes"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        if not space or space.type != "NODE_EDITOR":
            return False
        if _state.is_running:
            return False
        return space.edit_tree is not None

    def execute(self, context):
        tree = context.space_data.edit_tree
        prompt = context.scene.qna_user_question.strip()
        if not prompt:
            _state.error = "Please describe the node setup you want."
            _state.response_lines = []
            _state.raw_response = ""
            return {"CANCELLED"}

        # Reset all state
        _state.response_lines = []
        _state.raw_response = ""
        _state.error = ""
        _state.build_log = []
        _state.pending_phase1 = None
        _state.pending_phase2 = None
        _state.abort_requested = False
        _state.is_running = True
        _state.dirty = False
        _state.gen_phase = "nodes"

        # Snapshot tree state so the user can revert if generation goes wrong
        try:
            _state.pre_gen_snapshot = serialize_node_tree(tree)
        except Exception:
            _state.pre_gen_snapshot = None

        # Store context for multi-phase pipeline
        _state.gen_user_prompt = prompt
        _state.gen_base_url = _pref("ollama_base_url", "http://localhost:11434")
        _state.gen_model = context.scene.qna_model_generate or _pref(
            "ollama_model_generate", "qwen2.5-coder:14b"
        )
        _state.gen_tree_type = tree.bl_idname
        _state.loaded_base_url = _state.gen_base_url
        _state.loaded_model = _state.gen_model
        _state.model_is_loaded = True

        # Store clear preference in state so the timer callback can read it
        _state.gen_clear = context.scene.qna_clear_before_gen

        # Include existing tree context if not clearing
        existing_context = ""
        clear = _state.gen_clear
        if not clear and len(tree.nodes) > 0:
            try:
                existing = serialize_node_tree(tree)
                existing_context = (
                    "\n\nExisting nodes in the tree (extend this setup):\n"
                    f"```json\n{json.dumps(existing, indent=2)}\n```\n"
                )
            except Exception:
                pass

        full_prompt = prompt + existing_context

        # Launch step 1
        thread = threading.Thread(
            target=_bg_generate_step1,
            args=(
                _state.gen_base_url,
                _state.gen_model,
                full_prompt,
                _state.gen_tree_type,
            ),
            daemon=True,
        )
        thread.start()
        if not bpy.app.timers.is_registered(_poll_timer):
            bpy.app.timers.register(_poll_timer, first_interval=0.2)
        return {"FINISHED"}


class QNA_OT_CopyResponse(bpy.types.Operator):
    """Copy response to clipboard"""

    bl_idname = "qna.copy_response"
    bl_label = "Copy Response"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bool(_state.raw_response)

    def execute(self, context):
        context.window_manager.clipboard = _state.raw_response
        self.report({"INFO"}, "Copied.")
        return {"FINISHED"}


class QNA_OT_ExportTree(bpy.types.Operator):
    """Export node tree JSON to text block"""

    bl_idname = "qna.export_tree_json"
    bl_label = "Export Tree JSON"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space and space.type == "NODE_EDITOR" and space.edit_tree is not None

    def execute(self, context):
        tree = context.space_data.edit_tree
        try:
            data = serialize_node_tree(tree)
            name = f"NodeTree_{tree.name}.json"
            _save_debug_text(name, json.dumps(data, indent=2))
            self.report({"INFO"}, f"Exported to '{name}'")
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
        return {"FINISHED"}


# === PANELS ===


class _QNA_PT_Base:
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Qwen Assistant"

    def draw(self, context):
        _state.panel_last_drawn = time.time()
        layout = self.layout
        scene = context.scene
        space = context.space_data
        tree = space.edit_tree if space else None

        if tree:
            header = layout.row()
            header.label(text=f"Tree: {tree.name}", icon="NODETREE")
            sub = layout.row()
            sub.label(
                text=f"{len(tree.nodes)} nodes · {len(tree.links)} links",
                icon="INFO",
            )
        else:
            layout.label(text="No active node tree.", icon="ERROR")
            return

        layout.separator()

        row = layout.row(align=True)
        row.scale_y = 1.2
        row.prop(scene, "qna_mode", expand=True)
        layout.separator()

        mode = scene.qna_mode

        # --- Model selector ---
        model_row = layout.row(align=True)
        if mode == "ANALYZE":
            model_row.prop(scene, "qna_model_analyze", text="Model")
            selected_model = scene.qna_model_analyze
        else:
            model_row.prop(scene, "qna_model_generate", text="Model")
            selected_model = scene.qna_model_generate
        model_row.operator("qna.refresh_models", text="", icon="FILE_REFRESH")

        # --- Model description ---
        profile = _get_model_profile(selected_model)
        if profile:
            box = layout.box()
            box.scale_y = 0.75
            box.label(text=f"+ {profile['pros']}", icon="CHECKMARK")
            box.label(text=f"- {profile['cons']}", icon="X")
        elif selected_model and selected_model != "none":
            box = layout.box()
            box.scale_y = 0.75
            box.label(text="Unknown model — no profile available", icon="QUESTION")

        # --- Generate-only options ---
        if mode == "GENERATE":
            layout.prop(scene, "qna_clear_before_gen")

        layout.separator()

        if mode == "ANALYZE":
            layout.label(text="Your question:")
        else:
            layout.label(text="Describe the node setup you want:")
        layout.prop(scene, "qna_user_question", text="")

        row = layout.row(align=True)
        main_btn = row.row(align=True)
        main_btn.scale_y = 1.5

        if _state.is_running:
            main_btn.enabled = False
            phase = _state.gen_phase
            if phase == "nodes":
                label = "Step 1: Selecting nodes…"
            elif phase == "building":
                label = "Building nodes…"
            elif phase == "connecting":
                label = "Step 2: Connecting…"
            elif phase == "linking":
                label = "Creating links…"
            else:
                label = "Working…"
            main_btn.operator(
                "qna.ask_qwen" if mode == "ANALYZE" else "qna.generate_nodes",
                text=label,
                icon="SORTTIME",
            )
        elif mode == "ANALYZE":
            main_btn.operator("qna.ask_qwen", text="Ask Qwen", icon="VIEWZOOM")
        else:
            main_btn.operator(
                "qna.generate_nodes",
                text="Generate Nodes",
                icon="NODETREE",
            )

        row.operator("qna.export_tree_json", text="", icon="FILE_TEXT")
        row.operator("qna.copy_response", text="", icon="COPYDOWN")

        if _state.is_running:
            abort_row = layout.row()
            abort_row.alert = True
            abort_row.scale_y = 1.2
            abort_row.operator("qna.abort", text="Abort", icon="X")

        if _state.pre_gen_snapshot is not None and not _state.is_running:
            revert_row = layout.row()
            revert_row.scale_y = 1.1
            revert_row.operator(
                "qna.revert", text="Undo Last Generation", icon="LOOP_BACK"
            )

        layout.separator()

        if _state.error:
            box = layout.box()
            box.alert = True
            wrap_w = _pref("response_wrap_width", 62)
            for line in textwrap.wrap(_state.error, width=wrap_w):
                box.label(text=line, icon="ERROR")
            return

        if _state.response_lines or _state.is_running:
            box = layout.box()
            if _state.is_running and not _state.response_lines:
                box.label(text="Waiting for response…", icon="SORTTIME")
            else:
                for line in _state.response_lines:
                    box.label(text=line if line else " ")
            if _state.is_running:
                box.label(text="▍ working…", icon="SORTTIME")


class QNA_PT_ShaderPanel(_QNA_PT_Base, bpy.types.Panel):
    bl_idname = "QNA_PT_ShaderPanel"
    bl_label = "Qwen Assistant"

    @classmethod
    def poll(cls, context):
        s = context.space_data
        return s and s.type == "NODE_EDITOR" and s.tree_type == "ShaderNodeTree"


class QNA_PT_GeometryPanel(_QNA_PT_Base, bpy.types.Panel):
    bl_idname = "QNA_PT_GeometryPanel"
    bl_label = "Qwen Assistant"

    @classmethod
    def poll(cls, context):
        s = context.space_data
        return s and s.type == "NODE_EDITOR" and s.tree_type == "GeometryNodeTree"


class QNA_PT_CompositorPanel(_QNA_PT_Base, bpy.types.Panel):
    bl_idname = "QNA_PT_CompositorPanel"
    bl_label = "Qwen Assistant"

    @classmethod
    def poll(cls, context):
        s = context.space_data
        return s and s.type == "NODE_EDITOR" and s.tree_type == "CompositorNodeTree"


# === REGISTRATION ===

_classes = (
    QNA_AddonPreferences,
    QNA_OT_RefreshModels,
    QNA_OT_Abort,
    QNA_OT_Revert,
    QNA_OT_AskQwen,
    QNA_OT_GenerateNodes,
    QNA_OT_CopyResponse,
    QNA_OT_ExportTree,
    QNA_PT_ShaderPanel,
    QNA_PT_GeometryPanel,
    QNA_PT_CompositorPanel,
)


def _unload_model_bg(base_url: str, model: str):
    """Send keep_alive=0 to Ollama to evict the model from VRAM."""
    try:
        url = f"{base_url.rstrip('/')}/api/generate"
        payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
    except Exception:
        pass  # Ollama may not be running — ignore


_PANEL_UNLOAD_DELAY = 4.0  # seconds after panel closes before unloading model


def _panel_watch_timer() -> float:
    """
    Runs every 2 s. If the N-panel hasn't been drawn for PANEL_UNLOAD_DELAY seconds
    and no generation is in progress, unload the model from VRAM.
    """
    if (
        _state.model_is_loaded
        and not _state.is_running
        and _state.loaded_model
        and (time.time() - _state.panel_last_drawn) > _PANEL_UNLOAD_DELAY
    ):
        _state.model_is_loaded = False
        threading.Thread(
            target=_unload_model_bg,
            args=(_state.loaded_base_url, _state.loaded_model),
            daemon=True,
        ).start()
    return 2.0  # repeat every 2 seconds


def _auto_fetch_models():
    """Timer callback: silently populate model list on addon load."""
    try:
        p = _get_prefs()
        base_url = p.ollama_base_url if p else "http://localhost:11434"
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        all_names = sorted(m["name"] for m in data.get("models", []))
        usable = [n for n in all_names if not _is_embed_model(n)]
        if usable:
            _rebuild_model_lists(all_names)
    except Exception:
        pass  # Ollama not running — user can click Refresh manually
    return None  # do not repeat


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    # Silently try to populate models 0.5 s after load (non-blocking)
    bpy.app.timers.register(_auto_fetch_models, first_interval=0.5)
    # Watch panel visibility to unload model from VRAM when panel is closed
    bpy.app.timers.register(_panel_watch_timer, first_interval=2.0, persistent=True)

    bpy.types.Scene.qna_user_question = bpy.props.StringProperty(
        name="Prompt",
        description="Question or description for the LLM",
        default="",
    )
    bpy.types.Scene.qna_mode = bpy.props.EnumProperty(
        name="Mode",
        items=[
            (
                "ANALYZE",
                "Analyze",
                "Ask questions about the current node tree",
                "VIEWZOOM",
                0,
            ),
            (
                "GENERATE",
                "Generate",
                "Create nodes from a text description",
                "NODETREE",
                1,
            ),
        ],
        default="ANALYZE",
        update=_on_mode_change,
    )
    bpy.types.Scene.qna_model_generate = bpy.props.EnumProperty(
        name="Generate Model",
        description="Model used for node generation — list sorted best-for-generation first",
        items=_enum_generate_items,
    )
    bpy.types.Scene.qna_model_analyze = bpy.props.EnumProperty(
        name="Analyze Model",
        description="Model used for node tree analysis — list sorted best-for-analysis first",
        items=_enum_analyze_items,
    )
    bpy.types.Scene.qna_clear_before_gen = bpy.props.BoolProperty(
        name="Clear Tree Before Generating",
        description="Remove all existing nodes before placing generated ones",
        default=True,
    )


def unregister():
    if bpy.app.timers.is_registered(_poll_timer):
        bpy.app.timers.unregister(_poll_timer)
    if bpy.app.timers.is_registered(_panel_watch_timer):
        bpy.app.timers.unregister(_panel_watch_timer)
    for attr in (
        "qna_user_question",
        "qna_mode",
        "qna_model_generate",
        "qna_model_analyze",
        "qna_clear_before_gen",
    ):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
