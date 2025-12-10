import bpy
import socket
import json
import struct

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
BONE_NAME = "Head"  # Name of the head bone in your armature

class OpenVtuberOperator(bpy.types.Operator):
    """Real-time VTuber Receiver"""
    bl_idname = "wm.open_vtuber_receiver"
    bl_label = "OpenVtuber Receiver (Modal)"
    
    _timer = None
    _sock = None

    def modal(self, context, event):
        if event.type == 'TIMER':
            try:
                # Non-blocking receive
                data, addr = self._sock.recvfrom(1024)
                data = json.loads(data.decode('utf-8'))
                self.update_avatar(context, data)
            except socket.error:
                pass  # No data received this frame
            except Exception as e:
                print(f"Error: {e}")

        elif event.type in {'ESC'}:
            self.cancel(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def update_avatar(self, context, data):
        obj = context.object
        
        if not obj:
            return

        # 1. Handle Armature (Head Rotation)
        if obj.type == 'ARMATURE':
            pose_bone = obj.pose.bones.get(BONE_NAME)
            if pose_bone:
                # Map Pitch/Yaw/Roll to Euler angles (adjust indices based on your rig)
                # Blender uses Radians. Tracker sends Degrees.
                p = data["head"]["p"] * (3.14159 / 180.0)
                y = data["head"]["y"] * (3.14159 / 180.0)
                r = data["head"]["r"] * (3.14159 / 180.0)
                
                # Apply Rotation (Modify XYZ order if rig is different)
                pose_bone.rotation_mode = 'XYZ'
                pose_bone.rotation_euler = (p, r, -y) # Often Y is yaw, but check your rig axis

        # 2. Handle Mesh (Shape Keys)
        # If the armature has a child mesh, or if we selected the mesh
        mesh_obj = None
        if obj.type == 'MESH':
            mesh_obj = obj
        elif obj.children:
            for child in obj.children:
                if child.type == 'MESH' and child.data.shape_keys:
                    mesh_obj = child
                    break
        
        if mesh_obj and mesh_obj.data.shape_keys:
            keys = mesh_obj.data.shape_keys.key_blocks
            
            # Mouth
            if "MouthOpen" in keys:
                keys["MouthOpen"].value = data["mouth"]["open"]
            
            # Eyes (Blink)
            blink_val = data["eyes"]["blink"]
            if "Blink" in keys:
                keys["Blink"].value = blink_val
            if "EyesClosed" in keys:
                keys["EyesClosed"].value = blink_val
                
            # Brows
            brow_val = data["brows"]["raise"]
            if "BrowsUp" in keys:
                keys["BrowsUp"].value = brow_val

    def execute(self, context):
        # Setup Socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((UDP_IP, UDP_PORT))
        self._sock.setblocking(False)
        
        # Setup Timer (runs every 0.016s -> approx 60fps)
        self._timer = context.window_manager.event_timer_add(0.016, window=context.window)
        context.window_manager.modal_handler_add(self)
        
        self.report({'INFO'}, f"OpenVtuber Listening on {UDP_PORT}...")
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        if self._sock:
            self._sock.close()
        self.report({'INFO'}, "OpenVtuber Stopped.")

def register():
    bpy.utils.register_class(OpenVtuberOperator)

def unregister():
    bpy.utils.unregister_class(OpenVtuberOperator)

if __name__ == "__main__":
    register()
    # Trigger immediately for testing (optional)
    # bpy.ops.wm.open_vtuber_receiver()