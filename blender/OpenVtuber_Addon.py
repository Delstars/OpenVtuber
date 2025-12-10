bl_info = {
    "name": "OpenVtuber Receiver",
    "author": "OpenVtuber Team",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > OpenVtuber",
    "description": "One-click VTuber receiver",
    "category": "Animation",
}

import bpy
import socket
import json
import threading

# Global variables to handle socket thread
stop_event = None
sock = None
recv_thread = None
latest_data = {}

def server_loop(stop_event):
    global sock, latest_data
    udp_ip = "127.0.0.1"
    udp_port = 5005
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))
    sock.settimeout(0.1)
    
    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(1024)
            latest_data = json.loads(data.decode('utf-8'))
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error: {e}")
    sock.close()

class OpenVtuber_OT_Start(bpy.types.Operator):
    """Start Listening for Tracker"""
    bl_idname = "vtuber.start"
    bl_label = "Start Receiver"

    def execute(self, context):
        global stop_event, recv_thread
        if recv_thread and recv_thread.is_alive():
            self.report({'WARNING'}, "Already Running!")
            return {'CANCELLED'}
        
        stop_event = threading.Event()
        recv_thread = threading.Thread(target=server_loop, args=(stop_event,))
        recv_thread.daemon = True
        recv_thread.start()
        
        # Add modal timer to update UI from thread data
        bpy.ops.vtuber.modal_timer()
        self.report({'INFO'}, "OpenVtuber Started!")
        return {'FINISHED'}

class OpenVtuber_OT_Stop(bpy.types.Operator):
    """Stop Receiver"""
    bl_idname = "vtuber.stop"
    bl_label = "Stop Receiver"

    def execute(self, context):
        global stop_event
        if stop_event:
            stop_event.set()
        self.report({'INFO'}, "OpenVtuber Stopped.")
        return {'FINISHED'}

class OpenVtuber_OT_ModalTimer(bpy.types.Operator):
    bl_idname = "vtuber.modal_timer"
    bl_label = "OpenVtuber Update Loop"
    _timer = None

    def modal(self, context, event):
        global stop_event, latest_data
        
        if stop_event and stop_event.is_set():
            return self.cancel(context)

        if event.type == 'TIMER' and latest_data:
            # APPLY DATA TO AVATAR
            obj = context.active_object
            if obj and obj.type == 'ARMATURE':
                # Head Rotation
                pb = obj.pose.bones.get("Head")
                if pb:
                    h = latest_data.get("head", {})
                    pb.rotation_mode = 'XYZ'
                    pb.rotation_euler = (
                        h.get("p", 0) * 0.01745,
                        h.get("r", 0) * 0.01745,
                        -h.get("y", 0) * 0.01745
                    )
            
            # Shape Keys (Find mesh child)
            mesh = None
            if obj:
                 # Check if obj is mesh or has mesh children
                if obj.type == 'MESH': mesh = obj
                elif obj.children:
                    for c in obj.children:
                        if c.type == 'MESH' and c.data.shape_keys:
                            mesh = c
                            break
            
            if mesh and mesh.data.shape_keys:
                kb = mesh.data.shape_keys.key_blocks
                if "MouthOpen" in kb: kb["MouthOpen"].value = latest_data.get("mouth", {}).get("open", 0)
                if "Blink" in kb: kb["Blink"].value = latest_data.get("eyes", {}).get("blink", 0)

        return {'PASS_THROUGH'}

    def execute(self, context):
        self._timer = context.window_manager.event_timer_add(0.016, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        context.window_manager.event_timer_remove(self._timer)
        return {'CANCELLED'}

class OpenVtuber_PT_Panel(bpy.types.Panel):
    bl_label = "OpenVtuber"
    bl_idname = "VIEW3D_PT_openvtuber"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OpenVtuber'

    def draw(self, context):
        layout = self.layout
        layout.operator("vtuber.start", icon='PLAY')
        layout.operator("vtuber.stop", icon='PAUSE')
        layout.label(text="Select Armature first!")

classes = [OpenVtuber_OT_Start, OpenVtuber_OT_Stop, OpenVtuber_OT_ModalTimer, OpenVtuber_PT_Panel]

def register():
    for c in classes: bpy.utils.register_class(c)

def unregister():
    for c in classes: bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()