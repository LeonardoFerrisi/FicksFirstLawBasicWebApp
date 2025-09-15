"""
Streamlit Diffusion Simulation Demo
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import tempfile
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import from the updated diffusion_code module
from diffusion import (position_ions, ficks_law_1d, apply_boundary_conditions, 
                           draw_membrane_box, draw_compartments, _draw_box,
                           COLORS, run_plot as diffusion_run_plot, RNG as rng)

# ------- Constants -----------
# Define constants needed for the demo
ION_NAMES = ["Na+", "K+", "Ca2+", "Cl-"]
NUM_ION_TYPES = len(ION_NAMES)
D_constants = [0.133, 0.196, 0.079, 0.203]  # Scaled diffusion constants
box_size_3d = np.array([20.0, 10.0, 10.0])  # Same as in diffusion_code.py
colors = COLORS  # Reference the imported COLORS constant


# ------- Streamlit UI -----------
st.title("3D Diffusion Simulation Demo")
st.subheader("Simulating Ion Diffusion Across a Membrane using Fick's 1st Law")
st.markdown("*Note: No electrochemical gradient has been incorporated yet*")

num_ions_per_type = st.sidebar.slider("Ions per type", 10, 500, 100, 10)
num_compartments = st.sidebar.slider("Number of compartments (even)", 2, 20, 6, 2)
D_scaling = st.sidebar.slider("Diffusion constant scaling factor", 0.001, 1.0, 0.01, 0.001)
t_end = st.sidebar.slider("Simulation steps", 10, 2000, 100, 10)
show_compartments = st.sidebar.checkbox("Show Compartments", value=False)
dt = 1.0

membrane_thickness = 0.5
membrane_position = box_size_3d[0] / 2.0

# ------- Functions -----------

# ------- Simulation -----------
# Create concentration dictionaries for positioning
concentration_at_rest = dict(zip(ION_NAMES, [140.0, 4.0, 2.0, 110.0]))  # Extracellular
concentration_inside = dict(zip(ION_NAMES, [10.0, 140.0, 0.0001, 4.0]))  # Intracellular

# Position ions according to physiological concentrations
ion_positions, ion_types = position_ions(
    num_ions_per_type, 
    NUM_ION_TYPES, 
    ION_NAMES,
    box_size_3d, 
    membrane_position, 
    membrane_thickness,
    concentration_outside=concentration_at_rest,
    concentration_inside=concentration_inside
)

D = [d * D_scaling for d in D_constants]

# Run the simulation and store positions
positions_history = [ion_positions.copy()]
for t in range(1, t_end + 1):
    flux_diffusion = ficks_law_1d(
        ion_positions, 
        ion_types, 
        NUM_ION_TYPES, 
        box_size_3d,
        num_compartments, 
        DiffusionConst=D
    )
    ion_positions += dt * flux_diffusion
    ion_positions = apply_boundary_conditions(ion_positions, box_size_3d)
    positions_history.append(ion_positions.copy())

# ------- Plotting -----------
frame = st.slider("Frame", 0, t_end, 0, 1)
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
cmap = np.array(colors)[ion_types]
ax.scatter(positions_history[frame][:, 0], positions_history[frame][:, 1], positions_history[frame][:, 2], c=cmap, s=14, alpha=0.8, edgecolor='k', linewidths=0.2)
ax.set_xlim(0, box_size_3d[0])
ax.set_ylim(0, box_size_3d[1])
ax.set_zlim(0, box_size_3d[2])
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title(f"Time step: {frame}")

# Draw the membrane box
draw_membrane_box(ax, membrane_position, membrane_thickness, box_size_3d,
                  edge_color='black', face_color='gray', alpha=0.12, lw=2)

# Draw compartment boundaries if toggled
if show_compartments:
    draw_compartments(ax, box_size_3d, num_compartments)
st.pyplot(fig)


st.divider()

st.text("Made by Leonardo Ferrisi - 2025")