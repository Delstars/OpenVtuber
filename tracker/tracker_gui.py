import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import threading
import time
import math
import traceback

class OpenVtuberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenVtuber: Nuclear Stability 🛡️")
        self.root.geometry("320x350")
        
        self.running = False
        self.ip = "127.0.0.1"
        self.port = 5005

        # --- HISTORY FOR SMOOTHING ---
        self.alpha = 0.2 # Smoothing factor (0.1 = slow/smooth, 0.9 = fast/jittery)
        self.last_p, self.last_y, self.last_r = 0, 0, 0
        
        # --- GUI SETUP ---
        ttk.Label(root, text="OpenVtuber: Stable", font=("Segoe UI", 14, "bold")).pack(pady=10)
        self.status_label = ttk.Label(root, text="Status: Stopped", foreground="red")
        self.status_label.pack()

        frame_cam = ttk.Frame(root)
        frame_cam.pack(pady=5)
        ttk.Label(frame_cam, text="Cam ID:").pack(side='left')
        self.cam_input = ttk.Entry(frame_cam, width=5)
        self.cam_input.insert(0, "0")
        self.cam_input.pack(side='left', padx=5)

        # SLIDERS
        group = ttk.LabelFrame(root, text="Tuning")
        group.pack(padx=10, pady=5, fill="x")
        
        ttk.Label(group, text="Smoothing (Lower = Smoother)").pack(anchor='w')
        self.smooth_scale = tk.Scale(group, from_=0.01, to=0.5, resolution=0.01, orient='horizontal')
        self.smooth_scale.set(0.1)
        self.smooth_scale.pack(fill='x')
        
        self.btn_start = ttk.Button(root, text="START TRACKING", command=self.toggle_tracking)
        self.btn_start.pack(pady=10, ipady=5, fill='x', padx=20)
        ttk.Label(root, text="C: Calibrate | Q: Quit").pack(side='bottom', pady=5)

    def toggle_tracking(self):
        if not self.running:
            self.running = True
            self.btn_start.config(text="STOP TRACKING")
            self.status_label.config(text="Status: Active", foreground="green")
            threading.Thread(target=self.run_tracker, daemon=True).start()
        else:
            self.running = False
            self.btn_start.config(text="START TRACKING")
            self.status_label.config(text="Status: Stopping...", foreground="orange")

    def run_tracker(self):
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            try:
                cap = cv2.VideoCapture(int(self.cam_input.get()))
            except: 
                print("Error: Could not open camera.")
                self.stop_logic("Camera Error")
                return

            # --- NEW "FLAT" REFERENCE FACE (More Stable) ---
            # Used by standard VTube libraries to prevent flipping
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ], dtype="double")

            cal_p, cal_y, cal_r = 0, 0, 0
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
                        
                        # DRAW RED DOTS
                        for i in [1, 152, 33, 263]:
                            cx, cy = int(lm[i].x * img_w), int(lm[i].y * img_h)
                            cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

                        image_points = np.array([
                            (lm[1].x * img_w, lm[1].y * img_h),     # Nose
                            (lm[152].x * img_w, lm[152].y * img_h), # Chin
                            (lm[33].x * img_w, lm[33].y * img_h),   # Left Eye
                            (lm[263].x * img_w, lm[263].y * img_h), # Right Eye
                            (lm[61].x * img_w, lm[61].y * img_h),   # Left Mouth
                            (lm[291].x * img_w, lm[291].y * img_h)  # Right Mouth
                        ], dtype="double")

                        focal_length = img_w
                        center = (img_w / 2, img_h / 2)
                        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
                        dist_coeffs = np.zeros((4, 1))

                        # SOLVE
                        (success, rotation_vector, translation_vector) = cv2.solvePnP(
                            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                        )

                        rmat, _ = cv2.Rodrigues(rotation_vector)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                        # Raw angles
                        target_p = angles[0] * 360
                        target_y = angles[1] * 360
                        target_r = angles[2] * 360

                        # --- HARD CLAMP (The "Wall") ---
                        # If angles are physically impossible (> 50 degrees), Force them to limit
                        # This prevents the -64000 error completely.
                        target_p = max(-50, min(50, target_p))
                        target_y = max(-50, min(50, target_y))
                        target_r = max(-50, min(50, target_r))

                        # --- SIMPLE SMOOTHING ---
                        # Weighted Average: New = (Old * 0.9) + (New * 0.1)
                        alpha = self.smooth_scale.get()
                        self.last_p = (self.last_p * (1 - alpha)) + (target_p * alpha)
                        self.last_y = (self.last_y * (1 - alpha)) + (target_y * alpha)
                        self.last_r = (self.last_r * (1 - alpha)) + (target_r * alpha)

                        # Calibration
                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            cal_p, cal_y, cal_r = self.last_p, self.last_y, self.last_r
                            calibrated = True
                        
                        final_p = self.last_p - cal_p if calibrated else self.last_p
                        final_y = self.last_y - cal_y if calibrated else self.last_y
                        final_r = self.last_r - cal_r if calibrated else self.last_r

                        # Send Data
                        left_ear = (np.linalg.norm(np.array([lm[160].x, lm[160].y]) - np.array([lm[144].x, lm[144].y]))) * 10
                        mouth_dist = (lm[14].y - lm[13].y) * 100
                        
                        data = {
                            "head": {"p": final_p, "y": final_y, "r": final_r},
                            "eyes": {"blink": 1.0 if left_ear < 0.03 else 0.0},
                            "mouth": {"open": min(1.0, max(0.0, mouth_dist))},
                            "brows": {"raise": 0.0}
                        }
                        sock.sendto(json.dumps(data).encode(), (self.ip, self.port))
                        
                        # Visuals
                        cv2.putText(image, f"P: {int(final_p)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image, f"Y: {int(final_y)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image, f"R: {int(final_r)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow('OpenVtuber Stable', image)
                if cv2.waitKey(5) & 0xFF == ord('q'): break
            
            cap.release()
            cv2.destroyAllWindows()
            self.stop_logic("Stopped")
            
        except Exception as e:
            print("CRITICAL ERROR:")
            traceback.print_exc()
            self.stop_logic("Error!")

    def stop_logic(self, msg):
        self.running = False
        self.status_label.config(text=f"Status: {msg}", foreground="red")
        self.btn_start.config(text="START TRACKING")

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenVtuberApp(root)
    root.mainloop()