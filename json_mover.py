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
        elif id_type == 'NodeTree' and name in bpy.data.node_groups: return bpy.data.node_groups[name]
        return None
    return val

def extract_node_tree(node_tree, tree_type):
    data = {
        "tree_type": tree_type,
        "nodes": [],
        "links": []
    }
    
    for node in node_tree.nodes:
        props = {}
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

        data["nodes"].append({
            "name": node.name,
            "bl_idname": node.bl_idname,
            "location": list(node.location),
            "width": getattr(node, "width", 140.0),
            "height": getattr(node, "height", 100.0),
            "properties": props,
            "inputs": inputs_data,
            "outputs": outputs_data
        })
        
    for link in node_tree.links:
        data["links"].append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.identifier,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.identifier
        })
        
    return data

def build_node_tree(node_tree, data):
    node_tree.nodes.clear() # clear default nodes
    created_nodes = {}
    
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
        
        # properties
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
        
    # 2. Links
    for l_data in data.get("links", []):
        from_node = created_nodes.get(l_data["from_node"])
        to_node = created_nodes.get(l_data["to_node"])
        if from_node and to_node:
            from_sock_id = l_data["from_socket"]
            to_sock_id = l_data["to_socket"]
            
            f_sock = from_node.outputs.get(from_sock_id)
            if not f_sock:
                for sock in from_node.outputs:
                    if sock.identifier == from_sock_id:
                        f_sock = sock
                        break

            t_sock = to_node.inputs.get(to_sock_id)
            if not t_sock:
                for sock in to_node.inputs:
                    if sock.identifier == to_sock_id:
                        t_sock = sock
                        break
                        
            # fallback to names if identifier not matched exactly
            if not f_sock and from_sock_id in from_node.outputs.keys(): f_sock = from_node.outputs[from_sock_id]
            if not t_sock and to_sock_id in to_node.inputs.keys(): t_sock = to_node.inputs[to_sock_id]
                
            if f_sock and t_sock:
                node_tree.links.new(f_sock, t_sock)


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
            
        build_node_tree(node_tree, data)
        self.report({'INFO'}, f"Imported node tree {file_name} as {tree_type}")
        
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
        col.operator(NODE_OT_export_json_mover.bl_idname, text="Export Selected", icon='EXPORT')
        
        layout.separator()
        
        col = layout.column(align=True)
        col.label(text="Import JSON:")
        col.operator(NODE_OT_import_json_mover.bl_idname, text="Import to Window", icon='IMPORT')


class JSONMoverProps(PropertyGroup):
    selected_tree: EnumProperty(
        name="Node Tree",
        description="Select a Node Tree to Export",
        items=get_node_trees
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
