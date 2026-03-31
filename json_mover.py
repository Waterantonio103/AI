import bpy
import json
import os
from bpy_extras.io_utils import ExportHelper, ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy.types import Operator, Panel, PropertyGroup

bl_info = {
    "name": "JSON Mover Node Assistant",
    "author": "Antigravity",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Node IO",
    "description": "Export and Import Shader and Geometry Nodes to JSON",
    "category": "Node",
}

def get_node_trees(self, context):
    items = []
    
    def get_users_desc(item, item_type):
        users = []
        if item_type == 'MATERIAL':
            for obj in bpy.data.objects:
                if getattr(obj, "data", None) and hasattr(obj.data, "materials"):
                    if item.name in [m.name for m in obj.data.materials if m]:
                        col = obj.users_collection[0].name if obj.users_collection else "Unlinked"
                        users.append(f"{obj.name} ({col})")
        elif item_type == 'GEOMETRY':
            for obj in bpy.data.objects:
                for mod in getattr(obj, "modifiers", []):
                    if mod.type == 'NODES' and hasattr(mod, "node_group") and mod.node_group == item:
                        col = obj.users_collection[0].name if obj.users_collection else "Unlinked"
                        users.append(f"{obj.name} ({col})")
                        break
        if users:
            # limiting to 3 users to prevent huge tooltips
            desc_users = ", ".join(users[:3]) + ("..." if len(users) > 3 else "")
            return f"{item.name} -> {desc_users}"
        return f"{item.name} -> Unused"

    # Shader trees from materials
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            identifier = f"MAT|{mat.name}"
            name = mat.name
            description = get_users_desc(mat, 'MATERIAL')
            icon = 'NODE_MATERIAL' # Valid shader editor icon
            items.append((identifier, name, description, icon, len(items)))
            
    # Geometry Node groups 
    for ng in bpy.data.node_groups:
        if ng.type == 'GEOMETRY':
            identifier = f"GEO|{ng.name}"
            name = ng.name
            description = get_users_desc(ng, 'GEOMETRY')
            icon = 'NODETREE' # Valid geometry nodes icon
            items.append((identifier, name, description, icon, len(items)))
            
    if not items:
        items.append(("NONE", "No Node Trees", "No valid node trees", 'ERROR', 0))
        
    return items

def serialize_val(val):
    if isinstance(val, bpy.types.ID):
        return {"__datablock__": True, "id_type": type(val).__name__, "name": val.name}
    elif "Vector" in str(type(val)) or "Color" in str(type(val)) or "Euler" in str(type(val)):
        return list(val)
    elif type(val) in (int, float, str, bool, type(None)):
        return val
    elif isinstance(val, set):
        return list(val)
    elif hasattr(val, "name") and type(val).__name__ != 'type':
        return str(val.name)
    return None

def deserialize_val(val):
    if isinstance(val, dict) and val.get("__datablock__"):
        id_type = val.get("id_type")
        name = val.get("name")
        if id_type == 'Material' and name in bpy.data.materials: return bpy.data.materials[name]
        elif id_type == 'Object' and name in bpy.data.objects: return bpy.data.objects[name]
        elif id_type == 'Image' and name in bpy.data.images: return bpy.data.images[name]
        elif id_type == 'Collection' and name in bpy.data.collections: return bpy.data.collections[name]
        elif id_type == 'Texture' and name in bpy.data.textures: return bpy.data.textures[name]
        # NodeTree subclasses (GeometryNodeTree, ShaderNodeTree, CompositorNodeTree, etc.)
        # are serialised with their concrete type name, not the base class name 'NodeTree'.
        # Matching on 'NodeTree' in the name covers all of them.
        elif 'NodeTree' in (id_type or '') and name in bpy.data.node_groups:
            return bpy.data.node_groups[name]
        return None
    return val

def extract_node_tree(node_tree, tree_type, _shared_groups=None):
    """Recursively serialise *node_tree* and all sub-node-groups it references.

    *_shared_groups* is a flat dict ``{group_name: data}`` that is created once
    at the root call and shared across every recursive call so each sub-group is
    only exported once and circular references are avoided.
    """
    is_root = _shared_groups is None
    if is_root:
        _shared_groups = {}

    data = {
        "tree_type": tree_type,
        "nodes": [],
        "links": [],
        "interface": []
    }

    # Extract interface (Group Inputs/Outputs)
    if hasattr(node_tree, "interface"):
        for item in getattr(node_tree.interface, "items_tree", []):
            if getattr(item, "item_type", "SOCKET") == "SOCKET":
                data["interface"].append({
                    "name": item.name,
                    "in_out": item.in_out,
                    "socket_type": item.socket_type,
                    "identifier": getattr(item, "identifier", "")
                })

    for node in node_tree.nodes:
        props = {}
        if hasattr(node, "node_tree") and node.node_tree:
            sub_ng = node.node_tree
            s_val = serialize_val(sub_ng)
            if s_val is not None:
                props["node_tree"] = s_val
            # Recursively embed the sub-group so the target file can recreate it
            # even if it doesn't have it. Use _shared_groups as a visited set to
            # avoid infinite loops with self-referencing groups.
            if sub_ng.name not in _shared_groups:
                _shared_groups[sub_ng.name] = {}  # placeholder prevents re-entry
                sub_type = type(sub_ng).__name__  # e.g. 'GeometryNodeTree'
                _shared_groups[sub_ng.name] = extract_node_tree(
                    sub_ng, sub_type, _shared_groups
                )
                
        # Make sure to handle all custom properties without getting blocked by base properties
        for key in node.bl_rna.properties.keys():
            if key in ('name', 'type', 'location', 'dimensions', 'width', 'height', 'color', 'use_custom_color', 
                       'show_options', 'show_preview', 'show_texture', 'inputs', 'outputs'):
                continue
            prop = node.bl_rna.properties[key]
            if not prop.is_readonly:
                try:
                    val = getattr(node, key)
                    s_val = serialize_val(val)
                    if s_val is not None:
                        props[key] = s_val
                except:
                    pass
                    
        inputs_data = {}
        for idx, sock in enumerate(node.inputs):
            if not sock.is_linked and hasattr(sock, "default_value"):
                try:
                    val = getattr(sock, "default_value")
                    s_val = serialize_val(val)
                    if s_val is not None:
                        inputs_data[sock.identifier] = {
                            "index": idx,
                            "default_value": s_val
                        }
                except: pass

        outputs_data = {}
        for idx, sock in enumerate(node.outputs):
            try:
                if hasattr(sock, "default_value"):
                    val = getattr(sock, "default_value")
                    s_val = serialize_val(val)
                    if s_val is not None:
                        outputs_data[sock.identifier] = {
                            "index": idx,
                            "default_value": s_val
                        }
            except: pass

        n_data = {
            "name": node.name,
            "bl_idname": node.bl_idname,
            "location": list(node.location),
            "width": getattr(node, "width", 140.0),
            "height": getattr(node, "height", 100.0),
            "properties": props,
            "inputs": inputs_data,
            "outputs": outputs_data
        }

        # dynamic node items (Capture Attribute, Store Named Attribute, Repeat Zones)
        items_collections = ['capture_items', 'store_items', 'repeat_items', 'simulation_items']
        node_items = {}
        for col_name in items_collections:
            if hasattr(node, col_name):
                col = getattr(node, col_name)
                extracted_col = []
                for item in col:
                    item_data = {}
                    for k in item.bl_rna.properties.keys():
                        if k not in ('rna_type', 'name'):
                            try:
                                v = getattr(item, k)
                                sv = serialize_val(v)
                                if sv is not None:
                                    item_data[k] = sv
                            except: pass
                    extracted_col.append(item_data)
                if extracted_col:
                    node_items[col_name] = extracted_col
        if node_items:
            n_data["node_items"] = node_items

        data["nodes"].append(n_data)
        
    for link in node_tree.links:
        try:
            from_idx = list(link.from_node.outputs).index(link.from_socket)
        except ValueError:
            from_idx = None
        try:
            to_idx = list(link.to_node.inputs).index(link.to_socket)
        except ValueError:
            to_idx = None
        data["links"].append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.identifier,
            "from_socket_name": link.from_socket.name,
            "from_socket_index": from_idx,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.identifier,
            "to_socket_name": link.to_socket.name,
            "to_socket_index": to_idx,
        })

    if is_root:
        data["embedded_groups"] = _shared_groups

    return data

# Group Input/Output nodes have sockets that are dynamically mirrored from
# node_tree.interface. When interface sockets are recreated, Blender assigns
# brand-new identifiers, so the old identifiers stored in the JSON will no
# longer match. We resolve sockets with a priority chain:
#   1. Translated identifier via interface_id_map  (most reliable)
#   2. Unique name match                           (safe only when names are unique)
#   3. Positional index                            (definitive tie-breaker)
def _resolve_socket(collection, old_id, name, index, interface_id_map):
    """Return the best-matching socket from *collection*.

    Works correctly for both standard nodes and group boundary nodes
    (NodeGroupInput / NodeGroupOutput) whose identifiers change on every
    interface rebuild.
    """
    # 1. Try the mapped/translated identifier first
    mapped_id = interface_id_map.get(old_id, old_id)
    sock = collection.get(mapped_id)
    if sock:
        return sock

    # 2. Direct identifier lookup (in case map wasn't needed)
    if mapped_id != old_id:
        sock = collection.get(old_id)
        if sock:
            return sock

    # 3. Scan for identifier or unique name match
    name_matches = []
    for s in collection:
        if s.identifier == mapped_id or s.identifier == old_id:
            return s  # exact identifier hit
        if s.name == name:
            name_matches.append(s)

    if len(name_matches) == 1:
        return name_matches[0]  # unambiguous name match

    # 4. Positional index — definitive tie-breaker for duplicate names
    if index is not None and index < len(collection):
        return collection[index]

    # 5. Last resort: first name match even if ambiguous
    if name_matches:
        return name_matches[0]

    return None


def _ensure_embedded_groups(embedded):
    """Recreate sub-node-groups embedded in an exported JSON.

    Two-pass strategy:
      Pass 1 — Create EMPTY SHELLS for groups that don't exist yet, so every
               cross-reference can be resolved in Pass 2.  Groups that already
               exist in the target file are left completely untouched — their
               nodes, links, and interface stay exactly as-is so the outer
               GeometryNodeGroup node can bind to them without corruption.
      Pass 2 — Build the contents of only the newly-created shells.

    This prevents the .001 / .002 duplicate explosion that happens when we
    clear-and-rebuild groups the target file already has correctly set up.
    """
    # Pass 1: create shells that don't yet exist; record which ones we made
    to_build = []
    for group_name, group_data in embedded.items():
        if not group_data:          # skip empty placeholder entries
            continue
        if group_name not in bpy.data.node_groups:
            group_type = group_data.get("tree_type", "GeometryNodeTree")
            bpy.data.node_groups.new(name=group_name, type=group_type)
            to_build.append(group_name)
        # If it already exists → leave it alone; the node will bind to it

    # Pass 2: build only the shells we just created (all other shells exist)
    for group_name in to_build:
        ng = bpy.data.node_groups[group_name]
        build_node_tree(ng, embedded[group_name])


def build_node_tree(node_tree, data):
    node_tree.nodes.clear()  # clear default nodes
    created_nodes = {}

    # -1. Ensure any embedded sub-groups exist before we try to assign node.node_tree
    embedded = data.get("embedded_groups", {})
    if embedded:
        _ensure_embedded_groups(embedded)

    # 0. Recreate interface (needed for Group Input/Output nodes)
    interface_id_map = {}
    if hasattr(node_tree, "interface") and "interface" in data:
        node_tree.interface.clear()
        for idata in data["interface"]:
            try:
                new_sock = node_tree.interface.new_socket(name=idata["name"], in_out=idata["in_out"], socket_type=idata["socket_type"])
                if "identifier" in idata and idata["identifier"]:
                    interface_id_map[idata["identifier"]] = new_sock.identifier
            except Exception as e:
                print(f"Error creating interface socket {idata.get('name')}: {e}")
    
    # 1. Create nodes
    for n_data in data.get("nodes", []):
        try:
            node = node_tree.nodes.new(n_data["bl_idname"])
        except RuntimeError:
            print(f"Node type {n_data['bl_idname']} not supported or found. Skipping.")
            continue
            
        node.name = n_data["name"]
        node.location = n_data.get("location", (0,0))
        node.width = n_data.get("width", 140.0)
        
        # default socket values
        # dynamic interface items (Capture Attribute, Store Named Attribute, etc)
        # MUST do this before sockets are resolved so that dynamic sockets appear!
        for col_name, extracted_col in n_data.get("node_items", {}).items():
            if hasattr(node, col_name):
                col = getattr(node, col_name)
                col.clear() # reset default items
                for item_data in extracted_col:
                    # new() usually requires minimal args; name or type
                    # We try to use data_type and domain from the extracted data
                    dt = item_data.get('data_type', 'FLOAT')
                    name = item_data.get('name', '')
                    try:
                        new_item = col.new(dt, name)
                        for k, v in item_data.items():
                            if hasattr(new_item, k):
                                d_v = deserialize_val(v)
                                if d_v is not None:
                                    try: setattr(new_item, k, d_v)
                                    except: pass
                    except Exception as e:
                        print(f"Failed to create item in {col_name}: {e}")

        for k, v in n_data.get("properties", {}).items():
            if hasattr(node, k):
                try:
                    deserialized_val = deserialize_val(v)
                    if type(deserialized_val) == list and type(getattr(node, k)).__name__ in ('Color', 'Vector', 'Euler'):
                        setattr(node, k, deserialized_val)
                    elif deserialized_val is not None:
                        setattr(node, k, deserialized_val)
                except Exception as e:
                    print(f"Could not set {k} on {node.name}: {e}")
                    
        # default socket values
        for sock_id, sock_info in n_data.get("inputs", {}).items():
            idx = sock_info.get("index", 0)
            val = sock_info.get("default_value")
            
            deserialized_val = deserialize_val(val)
            
            target_sock = None
            if sock_id in node.inputs:
                target_sock = node.inputs[sock_id]
            elif idx < len(node.inputs):
                target_sock = node.inputs[idx]
                
            if target_sock and hasattr(target_sock, "default_value") and deserialized_val is not None:
                try:
                    if type(deserialized_val) == list: 
                        target_sock.default_value = deserialized_val
                    else:
                        target_sock.default_value = deserialized_val
                except: pass
                
        created_nodes[node.name] = node
        
    # 2a. Second-pass: retry node_tree assignment for GeometryNodeGroup nodes
    #     whose node_tree was None after Pass 1 (can happen if the group was
    #     created AFTER this node was processed in the node loop above, e.g.
    #     because embedded_groups weren't available for a nested sub-tree).
    for n_data in data.get("nodes", []):
        if n_data.get("bl_idname") != "GeometryNodeGroup":
            continue
        node = created_nodes.get(n_data["name"])
        if node is None or node.node_tree is not None:
            continue  # already resolved
        nt_ref = n_data.get("properties", {}).get("node_tree")
        if nt_ref:
            resolved = deserialize_val(nt_ref)
            if resolved is not None:
                try:
                    node.node_tree = resolved
                    print(f"[json_mover] Second-pass resolved node_tree "
                          f"'{resolved.name}' for '{node.name}'")
                except Exception as e:
                    print(f"[json_mover] Second-pass failed for '{node.name}': {e}")

    # 2. Links
    for l_data in data.get("links", []):
        from_node = created_nodes.get(l_data["from_node"])
        to_node = created_nodes.get(l_data["to_node"])
        if not (from_node and to_node):
            continue

        f_sock = _resolve_socket(
            from_node.outputs,
            l_data["from_socket"],
            l_data.get("from_socket_name", ""),
            l_data.get("from_socket_index"),
            interface_id_map,
        )
        t_sock = _resolve_socket(
            to_node.inputs,
            l_data["to_socket"],
            l_data.get("to_socket_name", ""),
            l_data.get("to_socket_index"),
            interface_id_map,
        )

        if f_sock and t_sock:
            try:
                node_tree.links.new(f_sock, t_sock)
            except Exception as e:
                print(f"Failed to link {from_node.name}:{f_sock.name} "
                      f"→ {to_node.name}:{t_sock.name}: {e}")
        else:
            missing = []
            if not f_sock:
                missing.append(f"output '{l_data.get('from_socket_name')}' "
                               f"on {l_data['from_node']}")
            if not t_sock:
                missing.append(f"input '{l_data.get('to_socket_name')}' "
                               f"on {l_data['to_node']}")
            print(f"Could not resolve socket(s): {', '.join(missing)}")

    return interface_id_map


class NODE_OT_export_json_mover(Operator, ExportHelper):
    """Export the currently selected node tree to JSON"""
    bl_idname = "node.export_json_mover"
    bl_label = "Export Node Tree"
    
    filename_ext = ".json"
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def invoke(self, context, event):
        sc_props = context.scene.json_mover_props
        fname = sc_props.export_filename.strip()
        
        if not fname:
            tree_sel = sc_props.selected_tree
            if tree_sel != "NONE":
                fname = tree_sel.split("|", 1)[1]
            else:
                fname = "node_tree"
                
        if not fname.endswith(".json"):
            fname += ".json"
            
        export_path = sc_props.default_export_path.strip()
        if export_path:
            import os
            dir_path = bpy.path.abspath(export_path)
            self.filepath = os.path.join(dir_path, fname)
        else:
            self.filepath = fname
            
        return super().invoke(context, event)
    
    def execute(self, context):
        sc_props = context.scene.json_mover_props
        tree_sel = sc_props.selected_tree
        if tree_sel == "NONE":
            self.report({'WARNING'}, "No node tree selected for export.")
            return {'FINISHED'}
            
        parts = tree_sel.split("|", 1)
        tree_type = parts[0]
        tree_name = parts[1]
        
        node_tree = None
        if tree_type == "MAT":
            mat = bpy.data.materials.get(tree_name)
            if mat and mat.use_nodes:
                node_tree = mat.node_tree
                t_type = "SHADER"
        elif tree_type == "GEO":
            ng = bpy.data.node_groups.get(tree_name)
            if ng:
                node_tree = ng
                t_type = "GEOMETRY"
                
        if not node_tree:
            self.report({'ERROR'}, "Could not find node tree!")
            return {'CANCELLED'}
            
        data = extract_node_tree(node_tree, t_type)
        
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)
            
        self.report({'INFO'}, f"Exported {tree_name} to {self.filepath}")
        return {'FINISHED'}


class NODE_OT_import_json_mover(Operator, ImportHelper):
    """Import a node tree from a JSON file"""
    bl_idname = "node.import_json_mover"
    bl_label = "Import Node Tree"
    
    filename_ext = ".json"
    filter_glob: StringProperty(
        default="*.json",
        options={'HIDDEN'},
        maxlen=255,
    )
    
    def execute(self, context):
        with open(self.filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                self.report({'ERROR'}, "Invalid JSON file")
                return {'CANCELLED'}
                
        tree_type = data.get("tree_type")
        file_name = os.path.basename(self.filepath).replace(".json", "")
        
        node_tree = None
        space_tree_type = 'ShaderNodeTree'
        
        if tree_type == "SHADER":
            mat = bpy.data.materials.new(name=file_name)
            mat.use_nodes = True
            node_tree = mat.node_tree
            space_tree_type = 'ShaderNodeTree'
            
        elif tree_type == "GEOMETRY":
            ng = bpy.data.node_groups.new(name=file_name, type='GeometryNodeTree')
            node_tree = ng
            space_tree_type = 'GeometryNodeTree'
            
        else:
            self.report({'ERROR'}, "Unknown or missing tree_type in JSON")
            return {'CANCELLED'}
            
        interface_id_map = build_node_tree(node_tree, data)
        self.report({'INFO'}, f"Imported node tree {file_name} as {tree_type}")
        
        target_obj = context.scene.json_mover_props.import_target_object
        if target_obj:
            bpy.ops.object.select_all(action='DESELECT')
            target_obj.select_set(True)
            context.view_layer.objects.active = target_obj
            
            if tree_type == 'SHADER':
                if hasattr(target_obj, "data") and hasattr(target_obj.data, "materials"):
                    if not target_obj.data.materials:
                        target_obj.data.materials.append(mat)
                    else:
                        target_obj.data.materials[target_obj.active_material_index] = mat
            elif tree_type == 'GEOMETRY':
                has_geomod = False
                for mod in target_obj.modifiers:
                    if mod.type == 'NODES':
                        # Preserve modifier input bindings using the identifier map
                        old_inputs = {}
                        for k in mod.keys():
                            if k not in ('_RNA_UI',):
                                old_inputs[k] = mod[k]
                                
                        mod.node_group = ng
                        has_geomod = True
                        
                        # Reapply inputs to the new identifiers
                        for old_id, val in old_inputs.items():
                            base_id = old_id
                            suffix = ""
                            if old_id.endswith("_use_attribute"):
                                base_id = old_id[:-14]
                                suffix = "_use_attribute"
                            elif old_id.endswith("_attribute_name"):
                                base_id = old_id[:-15]
                                suffix = "_attribute_name"
                                
                            new_base = interface_id_map.get(base_id, base_id)
                            new_key = new_base + suffix
                            mod[new_key] = val
                        break
                if not has_geomod:
                    gn_mod = target_obj.modifiers.new(name="GeometryNodes", type='NODES')
                    gn_mod.node_group = ng
        
        # Switch Area to Node Editor
        area = context.area
        if area.type != 'NODE_EDITOR':
            node_area = next((a for a in context.screen.areas if a.type == 'NODE_EDITOR'), None)
            if node_area:
                area = node_area
            else:
                view_areas = [a for a in context.screen.areas if a.type == 'VIEW_3D']
                if view_areas:
                    area = max(view_areas, key=lambda a: a.width * a.height)
                    area.type = 'NODE_EDITOR'
        
        if area.type == 'NODE_EDITOR':
            space = area.spaces.active
            space.tree_type = space_tree_type
            space.pin = True # Pin the material to easily view it!
            if tree_type == 'GEOMETRY':
                space.node_tree = node_tree
            elif tree_type == 'SHADER':
                space.node_tree = node_tree
                
        return {'FINISHED'}


class VIEW3D_PT_json_mover(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Node IO"
    bl_label = "JSON Mover"
    
    def draw(self, context):
        layout = self.layout
        sc_props = context.scene.json_mover_props
        
        col = layout.column(align=True)
        col.label(text="Export JSON:")
        col.prop(sc_props, "selected_tree", text="")
        col.prop(sc_props, "export_filename", text="Export As")
        col.prop(sc_props, "default_export_path", text="Path")
        col.operator(NODE_OT_export_json_mover.bl_idname, text="Export Selected", icon='EXPORT')
        
        layout.separator()
        
        col = layout.column(align=True)
        col.label(text="Import JSON:")
        col.prop(sc_props, "import_target_object", text="Apply To")
        col.operator(NODE_OT_import_json_mover.bl_idname, text="Import to Window", icon='IMPORT')


class JSONMoverProps(PropertyGroup):
    selected_tree: EnumProperty(
        name="Node Tree",
        description="Select a Node Tree to Export",
        items=get_node_trees
    )
    export_filename: StringProperty(
        name="Export As",
        description="Name of the file to save as (leave blank to use the node tree's name)",
        default=""
    )
    default_export_path: StringProperty(
        name="Export Path",
        description="Default folder to save the JSON file",
        subtype='DIR_PATH',
        default=""
    )
    import_target_object: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name="Apply To",
        description="Object to apply the imported node tree to (optional)"
    )


classes = (
    JSONMoverProps,
    NODE_OT_export_json_mover,
    NODE_OT_import_json_mover,
    VIEW3D_PT_json_mover,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.json_mover_props = bpy.props.PointerProperty(type=JSONMoverProps)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.json_mover_props

if __name__ == "__main__":
    register()
