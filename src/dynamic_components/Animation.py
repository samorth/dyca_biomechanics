import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from dynamic_components.config import settings

MARKER_LOC = settings.marker_loc
# MARKER_LOC = {
#     'r_foot': [[13,12],[13,11],[12,14],[11,14],[11,10],[12,10],[13,14],[14,10],[8,9],[9,10],[10,13],[11,8],[12,8]],
#     'r_leg': [[6,7],[4,5],[6,4],[6,5],[7,4],[7,5]],
#     'hip': [[0,3],[3,2],[2,1],[0,2],[3,1],[0,1]],
#     'l_foot': [[24,23],[24,22],[23,25],[22,25],[22,21],[23,21],[24,25],[25,21],[19,20],[21,22],[24,21],[20,21],[22,19],[23,19]],
#     'l_leg': [[17,18],[15,17],[16,17],[15,16],[18,15],[18,16]],
#     'torso': [[27,26],[27,28],[26,28]],
#     'l_arm': [[33,34],[34,35],[35,28]],
#     'r_arm': [[32,31],[31,30],[30,29],[32,26]]
# }

class MotionCaptureAnimation:
    def __init__(self, df: pd.DataFrame, interval: int = 50, title="Animation", draw_lines: bool = True,
                 xlims: tuple = None, ylims: tuple = None, zlims: tuple = None):

        self.df = df
        self.interval = interval
        self.custom_title = title
        self.draw_lines = draw_lines

        # If axis limits are not provided, compute dynamically
        if xlims is None or ylims is None or zlims is None:
            # Extract x, y, z values from DataFrame
            x_vals = df.iloc[:, 1::3]
            y_vals = df.iloc[:, 2::3]
            z_vals = df.iloc[:, 3::3]

            xmin, xmax = x_vals.min().min(), x_vals.max().max()
            ymin, ymax = y_vals.min().min(), y_vals.max().max()
            zmin, zmax = z_vals.min().min(), z_vals.max().max()

            pad_x = 0.05 * (xmax - xmin)
            pad_y = 0.05 * (ymax - ymin)
            pad_z = 0.05 * (zmax - zmin)

            self.xlims = (xmin - pad_x, xmax + pad_x)
            self.ylims = (ymin - pad_y, ymax + pad_y)
            self.zlims = (zmin - pad_z, zmax + pad_z)
        else:
            # Use provided limits directly
            self.xlims = xlims
            self.ylims = ylims
            self.zlims = zlims

        # Set up figure and axes
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(self.xlims)
        self.ax.set_ylim(self.ylims)
        self.ax.set_zlim(self.zlims)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(self.custom_title)

        # Compute aspect ratio to make axes equally scaled
        x_range = self.xlims[1] - self.xlims[0]
        y_range = self.ylims[1] - self.ylims[0]
        z_range = self.zlims[1] - self.zlims[0]
        max_range = max(x_range, y_range, z_range)

        x_scale = x_range / max_range
        y_scale = y_range / max_range
        z_scale = z_range / max_range

        self.ax.set_box_aspect([x_scale, y_scale, z_scale])

        # Adjust figure size based on aspect ratio
        base_size = 6  
        fig_width  = base_size * x_scale
        fig_height = base_size * y_scale
        self.fig.set_size_inches(fig_width, fig_height)

        # Initialize scatter plot (no data yet)
        self.scat = self.ax.scatter([], [], [])

        # Initialize line objects for connections if requested
        if self.draw_lines:
            self.lines = []
            for part, connections in MARKER_LOC.items():
                for connection in connections:
                    line, = self.ax.plot([], [], [], 'k-', lw=1)
                    self.lines.append((connection, line))
        else:
            self.lines = None

        self.anim = None

    def _update(self, frame: int):
        # Get data for the current frame
        row = self.df.iloc[frame]
        xs = row.iloc[1::3].values
        ys = row.iloc[2::3].values
        zs = row.iloc[3::3].values

        # Update scatter points
        self.scat._offsets3d = (xs, ys, zs)
        self.ax.set_title(f"{self.custom_title} - Frame {int(row['Frame'])}")
        
        # Update lines between connected markers
        if self.draw_lines and self.lines is not None:
            for connection, line in self.lines:
                idx1, idx2 = connection  # Marker indices
                x_line = [xs[idx1], xs[idx2]]
                y_line = [ys[idx1], ys[idx2]]
                z_line = [zs[idx1], zs[idx2]]
                line.set_data(x_line, y_line)
                line.set_3d_properties(z_line)
        
        self.fig.canvas.draw_idle()
        return self.scat,

    def start_animation(self):
        # Create and start the animation if not already created
        self.anim = FuncAnimation(self.fig, self._update, frames=len(self.df),
                                  interval=self.interval, blit=False)
        plt.show()

    def save_animation(self, filename: str, writer: str = 'pillow'):
        # Save the animation to file, creating it if necessary
        if self.anim is None:
            self.anim = FuncAnimation(self.fig, self._update, frames=len(self.df),
                                      interval=self.interval, blit=False)
        self.anim.save(filename, writer=writer)



