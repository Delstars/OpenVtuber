import cv2
import mediapipe as mp
import numpy as np
import socket
import json
import time

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
WEBCAM_ID = 0
DEBUG = True  # Set to False for headless performance

# --- MEDIAPIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# --- SOCKET SETUP ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- UTILS ---
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_aspect_ratio(eye_points, landmarks, w, h):
    # Vertical lines
    v1 = distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    v2 = distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    # Horizontal line
    h_dist = distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    return (v1 + v2) / (2.0 * h_dist)

# Calibration offsets
cal_pitch, cal_yaw, cal_roll = 0, 0, 0
calibrated = False

cap = cv2.VideoCapture(WEBCAM_ID)

print("--- OpenVtuber Tracker ---")
print(f"Streaming to {UDP_IP}:{UDP_PORT}")
print("Press 'C' to Calibrate (Zero Pose)")
print("Press 'Q' to Quit")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip and convert
    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    data = {}

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            lm_3d = []
            lm_2d = []

            # Head Pose Estimation Points (Nose, Chin, Eyes, Mouth)
            points_idx = [1, 199, 33, 263, 61, 291]
            
            for idx in points_idx:
                x, y = int(lm[idx].x * img_w), int(lm[idx].y * img_h)
                lm_2d.append([x, y])
                lm_3d.append([x, y, lm[idx].z * 3000]) # Pseudo-depth

            lm_2d = np.array(lm_2d, dtype=np.float64)
            lm_3d = np.array(lm_3d, dtype=np.float64)

            # Camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(lm_3d, lm_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            # Raw Angles
            pitch, yaw, roll = angles[0] * 360, angles[1] * 360, angles[2] * 360

            # Calibration Logic
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cal_pitch, cal_yaw, cal_roll = pitch, yaw, roll
                calibrated = True
                print("Calibrated Zero Pose.")

            # Apply Calibration
            if calibrated:
                pitch -= cal_pitch
                yaw -= cal_yaw
                roll -= cal_roll

            # --- FEATURES ---
            # 1. Blink (EAR)
            # Left Eye indices: 33, 160, 158, 133, 153, 144
            # Right Eye indices: 362, 385, 387, 263, 373, 380
            left_ear = get_aspect_ratio([33, 160, 158, 133, 153, 144], 
                                        [[l.x*img_w, l.y*img_h] for l in lm], img_w, img_h)
            right_ear = get_aspect_ratio([362, 385, 387, 263, 373, 380], 
                                         [[l.x*img_w, l.y*img_h] for l in lm], img_w, img_h)
            
            # 2. Mouth Open
            mouth_top = lm[13]
            mouth_bot = lm[14]
            mouth_open = distance([mouth_top.x, mouth_top.y], [mouth_bot.x, mouth_bot.y]) * 10 # Scale up

            # 3. Eyebrows (Simple height check relative to eye)
            # Compare brow midpoint to eye midpoint
            brow_y = (lm[65].y + lm[295].y) / 2
            eye_y = (lm[159].y + lm[386].y) / 2
            brow_raise = max(0, (eye_y - brow_y) * 10)

            # --- PACKET ---
            data = {
                "head": {"p": pitch, "y": yaw, "r": roll},
                "eyes": {"blink": 1.0 if (left_ear + right_ear)/2 < 0.25 else 0.0},
                "mouth": {"open": min(1.0, max(0.0, mouth_open))},
                "brows": {"raise": min(1.0, max(0.0, brow_raise))}
            }

            # UDP Send
            sock.sendto(json.dumps(data).encode(), (UDP_IP, UDP_PORT))

            if DEBUG:
                cv2.putText(image, f"P: {int(pitch)} Y: {int(yaw)} R: {int(roll)}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if DEBUG:
        cv2.imshow('OpenVtuber Tracker', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()