import bpy
import mathutils

def create_debug_empty(matrix, name="DBG", size=0.15, collection=None, display='PLAIN_AXES'):
    # Ensure matrix is a mathutils.Matrix
    if not isinstance(matrix, mathutils.Matrix):
        matrix = mathutils.Matrix(matrix)

    # Create the empty (this adds it to bpy.data.objects but not to any collection)
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = display
    empty.empty_display_size = size

    # Choose a collection to link into:
    #  - if user provided a collection object, use it
    #  - if a string was given, try to find a collection with that name
    #  - otherwise use the active scene.collection (top-level Collection)
    target_col = None
    if isinstance(collection, bpy.types.Collection):
        target_col = collection
    elif isinstance(collection, str):
        target_col = bpy.data.collections.get(collection)
    if target_col is None:
        target_col = bpy.context.scene.collection

    # Link the object into the chosen collection (safe link)
    if empty.name not in target_col.objects:
        try:
            target_col.objects.link(empty)
        except RuntimeError:
            # fallback: link to scene collection
            bpy.context.scene.collection.objects.link(empty)

    # Apply world transform
    empty.matrix_world = matrix

    # Make sure it's visible
    try:
        empty.hide_set(False)            # Blender 2.8+ API for per-object viewport hide
    except Exception:
        empty.hide = False               # fallback older attribute
    empty.hide_viewport = False
    empty.hide_render = False

    # Ensure the collection and its parents are visible in the view layer
    for coll in empty.users_collection:
        # unhide collection in viewport (if hide_viewport exists)
        if hasattr(coll, "hide_viewport"):
            coll.hide_viewport = False

    # Update view layer and select the object so it's easy to find
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action='DESELECT')
    empty.select_set(True)
    bpy.context.view_layer.objects.active = empty

    # Diagnostic prints
    print("Created:", empty.name)
    print("  in collections:", [c.name for c in empty.users_collection])
    print("  location:", tuple(empty.location))
    print("  display type:", empty.empty_display_type)
    return empty



def create_debug_empties_from_matrices(matrices, prefix="DBG", size=0.15, collection=None, display='PLAIN_AXES'):
    """
    Create an empty for each matrix in matrices (iterable of 4x4 matrices or lists).
    Returns list of created objects.
    """
    created = []
    for i, m in enumerate(matrices):
        name = f"{prefix}_{i}"
        obj = create_debug_empty(m, name=name, size=size, collection=collection, display=display)
        created.append(obj)
    return created


def debug_pose_bones_world(armature_object=None, prefix="bone_dbg", size=0.12, collection=None, display='PLAIN_AXES'):
    """
    Create empties that visualize each pose bone's world-space transform.

    - armature_object: bpy.types.Object of type 'ARMATURE'. Defaults to active object.
    - Note: pose_bone.matrix is in armature/object space; multiply by armature.matrix_world for world space.
    """
    if armature_object is None:
        armature_object = bpy.context.object

    if armature_object is None or armature_object.type != 'ARMATURE':
        raise RuntimeError("Select or pass an armature object (bpy.types.Object with type == 'ARMATURE').")

    created = []
    for pb in armature_object.pose.bones:
        # pose bone matrix is in armature (object) space; multiply by armature.matrix_world for world space
        bone_m_world = armature_object.matrix_world @ pb.matrix
        name = f"{prefix}_{pb.name}"
        created.append(create_debug_empty(bone_m_world, name=name, size=size, collection=collection, display=display))
    return created
