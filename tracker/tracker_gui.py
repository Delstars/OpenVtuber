import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import threading
import time

class OpenVtuberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenVtuber Tracker")
        self.root.geometry("300x250")
        
        # Configuration
        self.running = False
        self.ip = "127.0.0.1"
        self.port = 5005
        
        # UI Elements
        ttk.Label(root, text="OpenVtuber 🎥", font=("Arial", 16, "bold")).pack(pady=10)
        
        self.status_label = ttk.Label(root, text="Status: Stopped", foreground="red")
        self.status_label.pack(pady=5)

        ttk.Label(root, text="Camera Index:").pack()
        self.cam_input = ttk.Entry(root)
        self.cam_input.insert(0, "0")
        self.cam_input.pack(pady=5)

        self.btn_start = ttk.Button(root, text="START TRACKING", command=self.toggle_tracking)
        self.btn_start.pack(pady=10, ipady=10, fill='x', padx=20)
        
        ttk.Label(root, text="Press 'C' in camera window to Calibrate").pack(side='bottom', pady=10)

    def toggle_tracking(self):
        if not self.running:
            self.running = True
            self.btn_start.config(text="STOP TRACKING")
            self.status_label.config(text="Status: Running", foreground="green")
            self.thread = threading.Thread(target=self.run_tracker)
            self.thread.daemon = True
            self.thread.start()
        else:
            self.running = False
            self.btn_start.config(text="START TRACKING")
            self.status_label.config(text="Status: Stopping...", foreground="orange")

    def run_tracker(self):
        # --- TRACKER LOGIC (Same as before, wrapped) ---
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        try:
            cam_idx = int(self.cam_input.get())
            cap = cv2.VideoCapture(cam_idx)
        except:
            self.stop_logic("Camera Error")
            return

        cal_pitch, cal_yaw, cal_roll = 0, 0, 0
        calibrated = False

        while self.running and cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            img_h, img_w, _ = image.shape
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    lm_2d, lm_3d = [], []
                    points_idx = [1, 199, 33, 263, 61, 291]
                    
                    for idx in points_idx:
                        x, y = int(lm[idx].x * img_w), int(lm[idx].y * img_h)
                        lm_2d.append([x, y])
                        lm_3d.append([x, y, lm[idx].z * 3000])

                    lm_2d, lm_3d = np.array(lm_2d, dtype=np.float64), np.array(lm_3d, dtype=np.float64)
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(lm_3d, lm_2d, cam_matrix, dist_matrix)
                    rmat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                    pitch, yaw, roll = angles[0] * 360, angles[1] * 360, angles[2] * 360

                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        cal_pitch, cal_yaw, cal_roll = pitch, yaw, roll
                        calibrated = True

                    if calibrated:
                        pitch -= cal_pitch
                        yaw -= cal_yaw
                        roll -= cal_roll

                    # Blink & Mouth logic (Simplified for brevity)
                    left_ear = (np.linalg.norm(np.array([lm[160].x, lm[160].y]) - np.array([lm[144].x, lm[144].y]))) * 10
                    mouth_open = (lm[14].y - lm[13].y) * 100

                    data = {
                        "head": {"p": pitch, "y": yaw, "r": roll},
                        "eyes": {"blink": 1.0 if left_ear < 0.02 else 0.0}, # Simple threshold
                        "mouth": {"open": min(1.0, max(0.0, mouth_open))},
                        "brows": {"raise": 0.0} # Add brow logic if needed
                    }
                    sock.sendto(json.dumps(data).encode(), (self.ip, self.port))
                    
                    cv2.imshow('Camera Feed (Minimize if slow)', image)

            if cv2.waitKey(5) & 0xFF == ord('q'): break
        
        cap.release()
        cv2.destroyAllWindows()
        self.stop_logic("Stopped")

    def stop_logic(self, msg):
        self.running = False
        self.status_label.config(text=f"Status: {msg}", foreground="red")
        self.btn_start.config(text="START TRACKING")

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenVtuberApp(root)
    root.mainloop()