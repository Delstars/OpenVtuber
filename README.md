# OpenVtuber 🎥

**OpenVtuber** is a lightweight, high-performance, open-source VTuber pipeline. It separates heavy computer vision processing from 3D rendering to ensure maximum framerates.

- **Tracker (Python):** Uses MediaPipe Face Mesh to track head pose, eyes, mouth, and eyebrows, streaming data via UDP.
- **Renderer (Blender):** A lightweight modal operator that receives data and animates a rigged character in real-time without freezing the UI.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/OpenVtuber.git](https://github.com/yourusername/OpenVtuber.git)
   cd OpenVtuber

2. Install tracker dependencies
Navigate to the tracker folder and install the required Python libraries:

Bash

cd tracker

pip install -r requirements.txt

🚀 Setup & Usage
1. Blender Setup (The Receiver)
Open Blender and load your rigged character.

Go to the Scripting tab.

Open blender/blender_receiver.py and click Run Script.

Press F3, search for OpenVtuber Receiver (Modal), and click to start it.

Tip: Open Window → Toggle System Console to see connection status.

Avatar Requirements
Head bone: Must be named Head.

Shape keys:

MouthOpen (0.0–1.0)

Blink or EyesClosed (0.0–1.0)

BrowsUp or EyebrowsRaised (0.0–1.0)

The Blender receiver searches for these names on the active mesh and drives them based on the tracker’s data.

2. Tracker Setup (The Camera)
Connect your webcam.

From the tracker folder, run:

Bash

python tracker.py
(Optional) Add --debug to see the camera feed and mesh overlay (enabled by default in script).

The tracker will:

Detect facial landmarks.

Compute head pose, eye blink, mouth openness, and eyebrow raise.

Stream values via UDP to 127.0.0.1:5005.

🎮 Controls
C – Calibrate (Zero Pose) Sit comfortably, look at the screen, and press C. The current head rotation is stored as "Center," compensating for camera tilt or seating offset.

Q – Quit Stops the tracker and closes the window.

🟢 OBS Studio Configuration (Broadcasting)
To use OpenVtuber with OBS:

Blender Settings
Set Render Engine to Eevee in the Render Properties panel.

In World Properties, set the Background Color to pure green (#00FF00).

In Render Properties → Film, ensure Transparent is unchecked so the background is solid green.

Hide overlays in the viewport (click the Show Overlays button in the top-right corner).

OBS Settings
Add a Window Capture source targeting the Blender window.

Right-click the source → Filters → + (Effect Filters) → Chroma Key.

Select Green as the key color.

Adjust Similarity and Smoothness untill the green background disappears and only the avatar remains.

📄 License
MIT License. You are free to use, modify, and distribute this project.
