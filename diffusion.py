"""
EXP1 - BME 6003 - Simulation of Ionic Diffusion through a Membrane

Author: Leonardo Ferrisi

Changes:
- Added brownian motion to the diffusion simulation.
- Added option to save the simulation as an animation GIF.
- Added variable (even number) compartments for fick's law simulation
- Added option to visualize compartments.
- Added slider 
    - Added key press events to control the slider (left/right arrows).
- Updated docstring and added typehints
- Fixed ion positioning to be more accurate
- Added ion concentrations inside and outside of the cell
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Global Constants -----------
RNG                 = np.random.default_rng(0) # random number generator
COLORS:list[str]    = ['red','green','blue','cyan']



# ------- Functions -----------

def position_ions(n_per_type:int, n_types:int, ion_names:list[str], box:np.ndarray, mem_pos:float, mem_thick:float,
                  concentration_outside:dict[list[str]:list[int]], 
                  concentration_inside:dict[list[str]:list[int]]
                  ) -> None:
    '''
    Place ions in 3D space, split by membrane position and relative concentrations.
    Ions are placed randomly in y and z, and split in x based on membrane position and 
    relative concentrations inside vs outside
    '''

    xl0, xl1 = 0.0, mem_pos - mem_thick/2.0
    xr0, xr1 = mem_pos + mem_thick/2.0, box[0]
    if not (xl1 > xl0 and xr1 > xr0):
        raise ValueError("Membrane thickness/position invalid for box size.")

    # how many of each type go inside vs outside based on relative concentrations
    n_in_list = []
    for t in range(n_types):
        ion_name = ion_names[t]
        c_out = concentration_outside.get(ion_name, 0)
        c_in = concentration_inside.get(ion_name, 0)
        total_c = c_in + c_out
        if total_c > 0:
            frac_in = c_in / total_c
        else:
            frac_in = 0.5
        n_in_list.append(int(round(n_per_type * frac_in)))

    # build per-type position blocks (x split left/right; y,z uniform)
    pos_blocks = []
    for t, n_in in enumerate(n_in_list):
        n_out = n_per_type - n_in
        x = np.r_[RNG.uniform(xl0, xl1, n_in), RNG.uniform(xr0, xr1, n_out)]
        y = RNG.uniform(0, box[1], n_per_type)
        z = RNG.uniform(0, box[2], n_per_type) 
        pos_blocks.append(np.column_stack((x, y, z)))

    positions = np.vstack(pos_blocks)
    types = np.repeat(np.arange(n_types, dtype=int), n_per_type)
    return positions, types

# > Plotting Functions
def draw_membrane_box(ax:Axes3D, x_center:float, thickness:float, box:np.ndarray, edge_color:str='k', face_color:str='gray', alpha:float=0.12, lw:int=2) -> Axes3D:
    """
    Draws the membrane box in 3D space.

    Returns:
        Axes3D: The Axes3D object, now modified with the membrane drawn.
    """

    # Lx, Ly, Lz = box
    x0 = x_center - thickness/2.0
    x1 = x_center + thickness/2.0
    y0, y1 = 0.0, box[1]
    z0, z1 = 0.0, box[2]

    C = np.array([
        [x0,y0,z0],[x0,y0,z1],[x0,y1,z0],[x0,y1,z1],
        [x1,y0,z0],[x1,y0,z1],[x1,y1,z0],[x1,y1,z1]
    ])

    faces = [
        [0,1,3,2],   
        [4,5,7,6],   
        [0,1,5,4],   
        [2,3,7,6],   
        [0,2,6,4],   
        [1,3,7,5],   
    ]
    polys = [C[idx] for idx in faces]

    poly = Poly3DCollection(polys, facecolors=face_color, edgecolors='none', alpha=alpha)
    ax.add_collection3d(poly)

    edges = [
        (0,1),(0,2),(1,3),(2,3),      
        (4,5),(4,6),(5,7),(6,7),      
        (0,4),(1,5),(2,6),(3,7)       
    ]
    segs = [(C[i], C[j]) for (i,j) in edges]
    lc = Line3DCollection(segs, colors=[edge_color], linewidths=lw)
    ax.add_collection3d(lc)
    return ax

def _draw_box(ax:Axes3D, bounds:tuple, edge_color:str, face_color:str, alpha:float, lw:int) -> None:
    """
    Helper function to draw a 3D box.
    """
    x0, x1, y0, y1, z0, z1 = bounds
    pts = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])
    verts = [
        [pts[0], pts[1], pts[5], pts[4]], [pts[2], pts[3], pts[7], pts[6]],
        [pts[0], pts[3], pts[7], pts[4]], [pts[1], pts[2], pts[6], pts[5]],
        [pts[0], pts[1], pts[2], pts[3]], [pts[4], pts[5], pts[6], pts[7]]
    ]
    pc = Poly3DCollection(verts, facecolors=face_color, alpha=alpha)
    ax.add_collection3d(pc)

    segs = [
        (pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0]),
        (pts[4], pts[5]), (pts[5], pts[6]), (pts[6], pts[7]), (pts[7], pts[4]),
        (pts[0], pts[4]), (pts[1], pts[5]), (pts[2], pts[6]), (pts[3], pts[7])
    ]
    lc = Line3DCollection(segs, colors=[edge_color], linewidths=lw)
    ax.add_collection3d(lc)

def draw_compartments(ax:Axes3D, box_size:np.ndarray, num_compartments:int, edge_color:str='black', face_color:str='gray', alpha:float=0.05, lw:int=1) -> None:
    """
    Draws the compartment boundaries as transparent boxes.
    """
    box_width = box_size[0] / num_compartments
    for i in range(num_compartments):
        x_start = i * box_width
        bounds = (x_start, x_start + box_width, 0, box_size[1], 0, box_size[2])
        _draw_box(ax, bounds, edge_color, face_color, alpha, lw)

def ficks_law_1d(positions:np.ndarray, 
                 types:list[str], 
                 n_types:int, 
                 box_size_3d:np.ndarray,
                 num_compartments:int, 
                 DiffusionConst:float=0.05, 
                 use_brownian_motion:bool=True,
                 brownian_motion_YZ:bool=False,
                 dt:int=1.0) -> np.ndarray:
    '''
    Simulates the movement of particles across a membrane using Fick's Law in one dimension.
    This function calculates the velocity of particles based on the concentration gradient
    across a series of compartments, adding a random component to simulate diffusion.

    Returns:
        np.ndarray: shape (length ion_types*num_ions_per_type, 3). An array representing the velocity of each particle in 3D space.
    '''
    if num_compartments % 2 != 0:
        raise ValueError("Number of compartments must be an even number.")

    N = positions.shape[0]
    vel = np.zeros_like(positions)
    gradients = np.zeros(N)
    
    box_width = box_size_3d[0]
    comp_width = box_width / num_compartments
    boundaries = np.linspace(0, box_width, num_compartments + 1)

    # Ensure D is an array of length n_types
    if np.isscalar(DiffusionConst):
        DiffusionConst = [DiffusionConst] * n_types
    DiffusionConst = np.asarray(DiffusionConst)

    vel = np.zeros_like(positions)

    for t in range(n_types):
        # Get counts of ions in each compartment
        counts_per_comp = np.zeros(num_compartments)
        for i in range(num_compartments):
            in_comp = (positions[:, 0] >= boundaries[i]) & (positions[:, 0] < boundaries[i+1])
            counts_per_comp[i] = np.sum((types == t) & in_comp)

        # Calculate gradient for each compartment and assign to particles
        for i in range(num_compartments):
            if i == 0:
                # Forward difference for the first compartment
                grad = (counts_per_comp[i+1] - counts_per_comp[i]) / comp_width
            elif i == num_compartments - 1:
                # Backward difference for the last compartment
                grad = (counts_per_comp[i] - counts_per_comp[i-1]) / comp_width
            else:
                # Central difference for interior compartments
                grad = (counts_per_comp[i+1] - counts_per_comp[i-1]) / (2 * comp_width)
            
            # Assign gradient to particles based on their compartment
            in_comp_mask = (positions[:, 0] >= boundaries[i]) & (positions[:, 0] < boundaries[i+1]) & (types == t)
            gradients[in_comp_mask] = grad

    # Calculate deterministic drift velocity based on gradient with ion-specific diffusion constants
    drift_velocity = np.zeros(N)
    for t in range(n_types):
        ion_mask = (types == t)
        drift_velocity[ion_mask] = -DiffusionConst[t] * gradients[ion_mask]

    # Add a random component to the velocity for each particle to simulate diffusion
    # The strength of the random motion is also scaled by the diffusion coefficient D
    if use_brownian_motion:
        random_motion_x = np.zeros(N)
        for t in range(n_types):
            ion_mask = (types == t)
            random_motion_x[ion_mask] = RNG.normal(loc=0.0, scale=np.sqrt(2 * DiffusionConst[t] * dt), size=np.sum(ion_mask))
    else:
        random_motion_x = 0.0 # No random motion

    # Combine drift and random motion for the final velocity in the x-direction
    vel[:, 0] = drift_velocity + random_motion_x
    
    # Add random motion in y and z directions as well, so particles spread in 3D
    if use_brownian_motion and brownian_motion_YZ:
        for t in range(n_types):
            ion_mask = (types == t)
            vel[ion_mask, 1] = RNG.normal(loc=0.0, scale=np.sqrt(2 * DiffusionConst[t] * dt), size=np.sum(ion_mask))
            vel[ion_mask, 2] = RNG.normal(loc=0.0, scale=np.sqrt(2 * DiffusionConst[t] * dt), size=np.sum(ion_mask))

    return vel

def apply_boundary_conditions(positions:np.ndarray, box_size:np.ndarray) -> np.ndarray:
    """
    Applies reflective boundary conditions to keep particles within the box.
    If a particle's new position is outside the box, it is reflected back in,
    simulating a bounce.
    """
    for i in range(3): # For x, y, and z axes
        # Bounce off the lower boundary (e.g., x=0)
        low_mask = positions[:, i] < 0
        positions[low_mask, i] = -positions[low_mask, i]
        
        # Bounce off the upper boundary (e.g., x=20)
        high_mask = positions[:, i] > box_size[i]
        positions[high_mask, i] = 2 * box_size[i] - positions[high_mask, i]
        
    return positions

def run_plot(ion_types:list[str], 
             ion_positions:np.ndarray, 
             ion_names:list[str],
             num_compartments:int, 
             box_size_3d:np.ndarray, 
             membrane_position:float,
             membrane_thickness:float,
             visualize_compartments:bool=False, 
             use_brownian_motion:bool=True, 
             brownian_motion_YZ:bool=False,
             diffusion_scaling_factor:float=1.0,
             save_gif:bool=False, 
             t_end:int=2000,
             dt:int=1.0,
             gif_filename:str='diffusion.gif') -> None:
    """
    The primary function for this demo.

    Sets up the 3D axes, draws the membrane box, (optionally) compartment boundaries, 
    and runs the diffusion simulation.
    """
    # num ion types
    num_ion_types = len(ion_names)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    cmap = np.array(COLORS)[ion_types]
    scatter_plot = ax.scatter(ion_positions[:, 0],
                            ion_positions[:, 1],
                            ion_positions[:, 2],
                            c=cmap, s=14, alpha=0.8, edgecolor='k', linewidths=0.2)

    ax.set_xlim(0, box_size_3d[0])
    ax.set_ylim(0, box_size_3d[1])
    ax.set_zlim(0, box_size_3d[2])
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

    handles = [Line2D([0],[0], marker='o', linestyle='', color=col, label=name)
            for name, col in zip(ion_names, COLORS)]
    ax.legend(handles=handles, loc='upper right')

    draw_membrane_box(ax, membrane_position, membrane_thickness, box_size_3d,
                  edge_color='black', face_color='gray', alpha=0.12, lw=2)

    if visualize_compartments:
        draw_compartments(ax, box_size_3d, num_compartments)

    Lx, Ly, Lz = box_size_3d
    x_left_center  = 0.25 * Lx
    x_right_center = 0.75 * Lx
    y_top = 0.92 * Ly
    z_back = 0.92 * Lz

    ax.text(x_left_center,  y_top, z_back, "Intracellular",
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))
    ax.text(x_right_center, y_top, z_back, "Extracellular",
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7))

    plt.tight_layout()
    fig.subplots_adjust(top=0.9) # Add space at the top for the title


    # ------------------ run & display only chosen steps ------------------
    # Run the full simulation and store the history of positions
    positions_history = [ion_positions.copy()]
    for t in range(1, t_end + 1):
        print(f'Simulating step {t}/{t_end}...')
        
        scaled_diffusion = [d * diffusion_scaling_factor for d in D]

        flux_diffusion = ficks_law_1d(ion_positions, 
                                      ion_types, 
                                      num_ion_types, 
                                      box_size_3d,
                                      num_compartments, 
                                      DiffusionConst=scaled_diffusion, 
                                      use_brownian_motion=use_brownian_motion,
                                      brownian_motion_YZ=brownian_motion_YZ,
                                      dt=dt)
        
        ion_positions += dt * flux_diffusion
        ion_positions = apply_boundary_conditions(ion_positions, box_size_3d)
        positions_history.append(ion_positions.copy())

    if save_gif:
        # Note: You may need to install Pillow: pip install Pillow
        print(f"Saving animation to {gif_filename}...")
        
        def animate(frame):
            scatter_plot._offsets3d = (positions_history[frame][:, 0], positions_history[frame][:, 1], positions_history[frame][:, 2])
            ax.set_title(f'Time step: {frame}')
            return scatter_plot,

        ani = animation.FuncAnimation(fig, animate, frames=len(positions_history), interval=50, blit=False)
        ani.save(gif_filename, writer='pillow', fps=15)
        print("Done.")
    else:
        # ---- Render the SLIDER --------------

        # Adjust subplot to make room for slider
        fig.subplots_adjust(bottom=0.25)

        # Create slider axis
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        time_slider = Slider(
            ax=ax_slider,
            label='Time Step',
            valmin=0,
            valmax=t_end,
            valinit=0,
            valstep=1
        )

        # Create an update function
        def update(val):
            frame = int(time_slider.val)
            scatter_plot._offsets3d = (positions_history[frame][:, 0], positions_history[frame][:, 1], positions_history[frame][:, 2])
            ax.set_title(f'Time step: {frame}')
            fig.canvas.draw_idle()

        # Register the update function with the slider
        time_slider.on_changed(update)

        # Add key press event for slider control
        def on_key_press(event):
            if event.key == 'right':
                new_val = time_slider.val + 1
                if new_val > t_end:
                    new_val = t_end
                time_slider.set_val(new_val)
            elif event.key == 'left':
                new_val = time_slider.val - 1
                if new_val < 0:
                    new_val = 0
                time_slider.set_val(new_val)

        fig.canvas.mpl_connect('key_press_event', on_key_press)

        # Set initial title
        ax.set_title('Time step: 0')

        plt.show()

# ------- Main execution -----------
if __name__ == "__main__":

    # Steps you want to display
    T_END   = 2000
    DT      = 1.0

    # Constants
    D                   = [1.330, 1.960, 0.793, 2.030] # Diffusion constant(s) for the ions (https://www.aqion.de/site/diffusion-coefficients)
    BOX_SIZE_3D         = np.array([20.0, 10.0, 10.0])   
    ION_NAMES:list[str] = ["Na+", "K+", "Ca2+", "Cl-"] # Four primary ion types

    # variables
    NUM_IONS_PER_TYPE:int    = 50
    NUM_ION_TYPES:int        = len(ION_NAMES)
    MEMBRANE_THICKNESS:float = 0.5
    MEMBRANE_POSITION:float  = BOX_SIZE_3D[0] / 2.0

    extracellular_concentration_at_rest:list[float] = [140.0, 4.0, 2.0, 110.0] # in mM (from https://www.researchgate.net/figure/Estimated-concentration-of-ions-in-the-intracellular-and-extracellular-fluid-inside-the_tbl1_346125469)
    concentration_at_rest:dict[list[str]:list[int]] = dict(zip(ION_NAMES, extracellular_concentration_at_rest))

    # Intracellular concentrations (typical physiological values)
    concentration_values_inside:list[float] = [10.0, 140.0, 0.0001, 4.0] # in mM
    concentration_inside:dict[list[str]:list[int]] = dict(zip(ION_NAMES, concentration_values_inside))

    ion_positions, ion_types = position_ions(NUM_IONS_PER_TYPE, 
                                             NUM_ION_TYPES, 
                                             ION_NAMES,
                                             BOX_SIZE_3D, 
                                             MEMBRANE_POSITION, 
                                             MEMBRANE_THICKNESS,
                                             concentration_outside=concentration_at_rest,
                                             concentration_inside=concentration_inside)
    
    run_plot(ion_types, 
             ion_positions, 
             ion_names=ION_NAMES,
             num_compartments=4, 
             box_size_3d=BOX_SIZE_3D, 
             membrane_position=MEMBRANE_POSITION,
             membrane_thickness=MEMBRANE_THICKNESS,
             visualize_compartments=True, 
             use_brownian_motion=True, 
             brownian_motion_YZ=False,
             diffusion_scaling_factor=0.01,
             save_gif=False,
             t_end=T_END,
             dt=DT)