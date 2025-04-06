import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# Load the ant.xml file
model_path = "ant.xml"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"MuJoCo model file not found at: {model_path}")

# Load model and create data
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Launch viewer window
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("🚀 Viewer launched. Running simulation...")
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Random control input
        data.ctrl[:] = np.random.uniform(low=-1.0, high=1.0, size=model.nu)

        mujoco.mj_step(model, data)

        # Sync viewer
        viewer.sync()

        # Slow down to real-time
        time.sleep(max(0.01 - (time.time() - step_start), 0))
