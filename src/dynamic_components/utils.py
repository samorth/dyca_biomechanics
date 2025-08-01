import numpy as np
import pandas as pd
import warnings
import re
import matplotlib.pyplot as plt
from dynamic_components.config import settings
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import logging
logger = logging.getLogger(__name__)


def get_filenames(data_dir):
    filenames = []
    for path in data_dir.iterdir():
        if path.is_file() and path.suffix == '.csv':
            filenames.append(path.name)
    
    def sort_key(name):
        match = re.search(r"Sub(\d+)_.*_T(\d+)", name)
        if match:
            sub_num = int(match.group(1))
            trial_num = int(match.group(2))
            return (sub_num, trial_num)
        else:
            return (float('inf'), float('inf')) 

    filenames.sort(key=sort_key)
    return filenames


def get_raw_data_columns(path):
    markers = pd.read_csv(path, sep=',', header=None, skiprows=2, nrows=2).to_numpy()
    name_markers = [markers[1,0], markers[1,1]]
    for idx in range(2,len(markers[0])):
        if pd.isna(markers[1,idx]):
            continue
        if pd.notna(markers[0,idx]):
            name = markers[0,idx].split(':')[1]
        name_markers.append(name + '_' + markers[1,idx])
    return name_markers


def check_for_faulty_channels(df):
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        missing_frac = df[col].isna().mean()
        if missing_frac > 0.05:
            warnings.warn(
                f"Spalte '{col}' enthält {missing_frac:.1%} fehlende Werte (>5%).", 
                UserWarning
            )
            logger.warning(f"Spalte '{col}' enthält {missing_frac:.1%} fehlende Werte (>5%).")
    return None


def get_signal_derivative(signal: np.ndarray, time_signal: np.ndarray):
    max = len(signal)
    dt = time_signal[1] - time_signal[0]
    derivative_signal = (signal[2:max, :] - signal[0:max - 2, :]) / (2 * dt)
    first = (signal[1, :] - signal[0, :]) / dt
    last = (signal[max - 1, :] - signal[max - 2, :]) / dt
    derivative_signal = np.vstack((first, derivative_signal, last))
    
    return derivative_signal


def get_95_criterion(eigenvalues):
    return np.sum(eigenvalues >= 0.95)


def get_steepest_slope(eigenvalues, return_curve=False):
    from scipy.signal import savgol_filter
    n = len(eigenvalues)
    polyorder = 2

    # Dynamische Wahl der Fensterlänge für den SG-Filter
    if n >= 100:
        window_length = 11
    else:
        wl = int(round(n / 10))
        if wl % 2 == 0:
            wl += 1
        window_length = max(wl, polyorder + 1)
        if window_length > n:
            window_length = n if n % 2 == 1 else n - 1

    # 1) Glatt abgeleitete Kurve
    if window_length <= polyorder or window_length > n:
        deriv_smooth = np.gradient(eigenvalues)
    else:
        deriv_smooth = savgol_filter(
            eigenvalues,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1
        )

    # 2) rohe diskrete Differenzen
    diffs = np.diff(eigenvalues)

    # 3) grobe Lokalisierung: Index minimaler glatter Ableitung
    j = int(np.argmin(deriv_smooth))

    # 4) Suche im Diferenz-Array in einem engen Fenster um j
    half_win = window_length // 2
    start = max(0, j - half_win)
    end   = min(len(diffs), j + half_win + 1)  # +1, weil slice exklusiv endet

    local_window = diffs[start:end]
    # Index des stärksten Einzelabfalls innerhalb des Fensters
    local_min_idx = int(np.argmin(local_window))

    # 5) Endgültiger Index: Position VOR dem Sprung
    idx_before_drop = start + local_min_idx

    if return_curve:
        return idx_before_drop, deriv_smooth
    return idx_before_drop

#marker_loc = settings.marker_loc
def plot_freeze_frame(
    df,
    frame_index,
    ax=None,
    title_prefix="Frame",
    draw_lines=True,
    elev=30,
    azim=-60,
    pad=0.0,
    trail_start=None,
    trail_lw=0.5,            # Linienbreite der Trajektorie
    trail_alpha=0.7,         # Transparenz
    trail_color='orange'
    
):
    """
    Plots a single freeze frame with dynamic equal scaling, adjustable bounds based on data,
    ticks starting from zero in ±250 increments, and optional padding.

    Parameters:
    - df: pandas DataFrame with marker coordinates in columns [Frame, x1, y1, z1, x2, y2, z2, ...]
    - frame_index: integer index of the frame to plot
    - ax: optional mpl 3D axis; if None, a new figure and axis are created
    - title_prefix: prefix for the subplot title
    - draw_lines: whether to connect markers by lines based on marker_loc
    - elev: elevation angle for the view
    - azim: azimuth angle for the view
    - pad: fraction of extra space to add around data bounds (e.g., 0.1 for 10%)
    - marker_loc: dict mapping connection lists for drawing lines (e.g., settings.marker_loc)
    """
    marker_loc = settings.marker_loc
    plt.style.use('ggplot')
    
    # Extract coords for the frame
    row = df.iloc[frame_index]
    xs = row.iloc[1::3].astype(float).values
    ys = row.iloc[2::3].astype(float).values
    zs = row.iloc[3::3].astype(float).values

    # Compute data bounds and padding
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()
    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min
    pad_x = span_x * pad
    pad_y = span_y * pad
    pad_z = span_z * pad

    # Determine axis limits based on data
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)
    z_lim = (z_min - pad_z, z_max + pad_z)

    # Setup figure and axis
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True
    else:
        fig = None
        
    #     # make the 3D panes fully transparent
    # ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    # # remove pane edges
    # ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    r, g, b, a = (229, 229, 229, 255)
    pane_face   = (r/255.0, g/255.0, b/255.0, a/255.0)   # RGBA
    pane_edge   = "#e5e5e5"              # dark blue hex string
    pane_width  = 0                    # edge line width in points

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        # turn on face filling and edge drawing
        axis.pane.set_facecolor(pane_face)
        axis.pane.set_edgecolor(pane_edge)
        axis.pane.set_linewidth(pane_width)

    # Apply limits and aspect
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect((x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]))

    # Scatter points
    ax.scatter(xs, ys, zs, marker='o', alpha=1)

    # Ticks starting at zero in ±250 increments
    def generate_ticks(lim):
        lower, upper = lim
        # positive ticks from 0 to ceiling(upper/250)*250
        pos_max = np.ceil(upper / 250.0) * 250.0
        ticks_pos = np.arange(0, pos_max + 1e-6, 250)
        # negative ticks from 0 to floor(lower/250)*250
        neg_min = np.floor(lower / 250.0) * 250.0
        ticks_neg = np.arange(0, neg_min - 1e-6, -250)
        return np.unique(np.concatenate((ticks_neg, ticks_pos)))

    #ax.set_xticks(generate_ticks(x_lim))
    #ax.set_yticks(generate_ticks(y_lim))
    #ax.set_zticks(generate_ticks(z_lim))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Minimalist style
    ax.grid(False)
    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

    # Title and view
    #label = int(row['Frame']) if 'Frame' in df.columns else frame_index
    #ax.set_title(f"{title_prefix} {label}")
    ax.view_init(elev=elev, azim=azim)
    
    #Markertrajektorien visualisieren
    if trail_start is not None and trail_start < frame_index:
        trail_df = df.iloc[trail_start:frame_index+1]
        xs_trail = trail_df.iloc[:, 1::3].astype(float).values
        ys_trail = trail_df.iloc[:, 2::3].astype(float).values
        zs_trail = trail_df.iloc[:, 3::3].astype(float).values
        num_markers = xs_trail.shape[1]
        for mi in range(num_markers):
            ax.plot(
                xs_trail[:, mi],
                ys_trail[:, mi],
                zs_trail[:, mi],
                lw=trail_lw,
                alpha=trail_alpha,
                color=trail_color
            )

    # Draw connecting lines if provided
    if draw_lines and marker_loc is not None:
        for conns in marker_loc.values():
            for i1, i2 in conns:
                if i1 < len(xs) and i2 < len(xs):
                    ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]], [zs[i1], zs[i2]], 'k-', lw=1)

    # Return figure if created
    if created_fig:
        return fig


def plot_freeze_frame_v2(
    df,
    frame_index,
    ax=None,
    title_prefix="Frame",
    draw_lines=True,
    elev=30,
    azim=-60,
    pad=0.0,
    trail_start=None,
    trail_lw=0.5,
    trail_alpha=0.7,
    trail_color='orange'
):
    """
    Plots a single freeze frame with dynamic equal scaling, adjustable bounds based on data,
    a minimalistic but visible bounding box styled in ggplot, and no axis ticks.

    Parameters:
    - df: pandas DataFrame with marker coordinates in columns [Frame, x1, y1, z1, x2, y2, z2, ...]
    - frame_index: integer index of the frame to plot
    - ax: optional mpl 3D axis; if None, a new figure and axis are created
    - title_prefix: prefix for the subplot title
    - draw_lines: whether to connect markers by lines based on marker_loc
    - elev: elevation angle for the view
    - azim: azimuth angle for the view
    - pad: fraction of extra space to add around data bounds (e.g., 0.1 for 10%)
    - trail_start: starting frame index for drawing a trajectory trail
    - trail_lw, trail_alpha, trail_color: style parameters for the trajectory trail
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    marker_loc = settings.marker_loc
    plt.style.use('ggplot')

    # Extract coords for the frame
    row = df.iloc[frame_index]
    xs = row.iloc[1::3].astype(float).values
    ys = row.iloc[2::3].astype(float).values
    zs = row.iloc[3::3].astype(float).values

    # Compute data bounds and padding
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()
    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min
    pad_x = span_x * pad
    pad_y = span_y * pad
    pad_z = span_z * pad

    # Determine axis limits
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)
    z_lim = (z_min - pad_z, z_max + pad_z)

    # Setup figure and axis
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True
    
    # Minimalistic but visible bounding box
    # Transparent faces, subtle edge lines
    pane_face = (1.0, 1.0, 1.0, 0.0)
    pane_edge = '#cccccc'
    pane_width = 1.0

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_face)
        axis.pane.set_edgecolor(pane_edge)
        axis.pane.set_linewidth(pane_width)

    # Apply limits and equal aspect
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect((x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]))

    # Scatter points
    ax.scatter(xs, ys, zs, marker='o', alpha=1)

    # Remove all ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # No grid lines
    ax.grid(False)

    # Set view
    ax.view_init(elev=elev, azim=azim)

    # Draw trajectory trail if requested
    if trail_start is not None and trail_start < frame_index:
        trail_df = df.iloc[trail_start:frame_index+1]
        xs_trail = trail_df.iloc[:, 1::3].astype(float).values
        ys_trail = trail_df.iloc[:, 2::3].astype(float).values
        zs_trail = trail_df.iloc[:, 3::3].astype(float).values
        for mi in range(xs_trail.shape[1]):
            ax.plot(
                xs_trail[:, mi],
                ys_trail[:, mi],
                zs_trail[:, mi],
                lw=trail_lw,
                alpha=trail_alpha,
                color=trail_color
            )

    # Draw connecting lines if provided
    if draw_lines and marker_loc is not None:
        for conns in marker_loc.values():
            for i1, i2 in conns:
                if i1 < len(xs) and i2 < len(xs):
                    ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]], [zs[i1], zs[i2]], 'k-', lw=1)

    if created_fig:
        return fig


def plot_freeze_frame_v3(
    df,
    frame_index,
    ax=None,
    title_prefix="Frame",
    draw_lines=True,
    elev=30,
    azim=-60,
    pad=0.0,
    trail_start=None,
    trail_lw=0.5,
    trail_alpha=0.7,
    trail_color='orange'
):
    """
    Plots a single freeze frame inside an inset 3D axis, allowing vertical padding
    between the white marker-box and the gray ggplot background.

    Parameters:
    - df: pandas DataFrame with marker coordinates (Frame, x1, y1, z1, ...)
    - frame_index: frame index to plot
    - ax: optional existing mpl 3D axis; if provided, an inset axis is created
    - title_prefix, draw_lines, elev, azim, pad, trail_start, trail_lw, trail_alpha, trail_color: as before
    """
    marker_loc = settings.marker_loc
    plt.style.use('ggplot')

    # Extract coords for the frame
    row = df.iloc[frame_index]
    xs = row.iloc[1::3].astype(float).values
    ys = row.iloc[2::3].astype(float).values
    zs = row.iloc[3::3].astype(float).values

    # Compute data bounds and padding
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()
    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min
    pad_x = span_x * pad
    pad_y = span_y * pad
    pad_z = span_z * pad

    # Determine axis limits
    x_lim = (x_min - pad_x, x_max + pad_x)
    y_lim = (y_min - pad_y, y_max + pad_y)
    z_lim = (z_min - pad_z, z_max + pad_z)

    created_fig = False

    if ax is not None:
        # Create an inset axis for the 3D box inside the gray background
        fig = ax.get_figure()
        # Hide original axis lines (gray pane remains)
        ax.set_axis_off()

        # Inset configuration: adjust bbox_to_anchor to control padding
        # Format: (x0, y0, width, height) in ax.transAxes coords
        # - Increase y0 to move inset up (more bottom padding)
        # - Decrease height to add top padding
        bottom_pad = 0.1  # fraction of ax height as bottom padding
        top_pad    = 0.1  # fraction of ax height as top padding

        inset_bbox = (0,                 # x0: left edge (0 = align to left)
                      bottom_pad,        # y0: start of inset above bottom
                      1.0,               # width: full width of parent ax
                      1.0 - bottom_pad - top_pad)  # height: reduce by pads

        ax_in = inset_axes(
            ax,
            width="100%", height="100%",
            bbox_to_anchor=inset_bbox,
            bbox_transform=ax.transAxes,
            loc='lower left', borderpad=0
        )
        ax = fig.add_axes(ax_in.get_position(), projection='3d')
        created_fig = True

    else:
        # Create a new figure and axis normally
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        created_fig = True

    # Style the minimalistic bounding box panes
    pane_face = (1.0, 1.0, 1.0, 0.0)
    pane_edge = '#cccccc'
    pane_width = 1.0
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_face)
        axis.pane.set_edgecolor(pane_edge)
        axis.pane.set_linewidth(pane_width)

    # Apply axis limits and equal aspect
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_box_aspect((x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]))

    # Plot the markers
    ax.scatter(xs, ys, zs, marker='o', alpha=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)

    # Optional trail
    if trail_start is not None and trail_start < frame_index:
        trail_df = df.iloc[trail_start:frame_index+1]
        xs_trail = trail_df.iloc[:,1::3].astype(float).values
        ys_trail = trail_df.iloc[:,2::3].astype(float).values
        zs_trail = trail_df.iloc[:,3::3].astype(float).values
        for mi in range(xs_trail.shape[1]):
            ax.plot(xs_trail[:,mi], ys_trail[:,mi], zs_trail[:,mi],
                    lw=trail_lw, alpha=trail_alpha, color=trail_color)

    # Connecting lines
    if draw_lines and marker_loc is not None:
        for conns in marker_loc.values():
            for i1, i2 in conns:
                if i1 < len(xs) and i2 < len(xs):
                    ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]], [zs[i1], zs[i2]],
                            'k-', lw=1)

    if created_fig:
        return fig
    
def plot_modes(
    mode_vector,
    column_names,
    ax=None,
    title="",
    xlabel="",
    ylabel="Mode size per marker \n [arbitrary unit]",
    tick_step=1
):
    import math
    """
    Plots average mode values per marker with customizable labels and fine grid.

    Parameters
    ----------
    mode_vector : array-like
        1D-Array der Modus-Werte, Länge muss len(column_names) entsprechen.
    column_names : list of str
        Spaltennamen, erwartet wird 'Marker_X', 'Marker_Y', ... – der Teil vor '_' wird als Marker
        gruppiert.
    ax : matplotlib.axes.Axes, optional
        Ziel-Achse. Wenn None, wird eine neue Figure mit Axes (12×6) erzeugt.
    title : str, optional
        Plot-Titel.
    xlabel : str, optional
        Beschriftung der x-Achse.
    ylabel : str, optional
        Beschriftung der y-Achse.
    tick_step : int or float, optional
        Abstand der yticks (Standard 250).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Achse mit dem erzeugten Balkendiagramm.
    """
    mv = np.array(mode_vector).squeeze().astype(float)
    if mv.ndim != 1 or mv.shape[0] != len(column_names):
        raise ValueError(
            f"Array-Länge {mv.shape[0]} ≠ Anzahl Spaltennamen {len(column_names)}"
        )

    # Werte nach Marker gruppieren
    marker_vals = {}
    for name, val in zip(column_names, mv):
        marker = name.split("_", 1)[0]
        marker_vals.setdefault(marker, []).append(val)

    # Mittelwerte berechnen
    means = {m: float(np.mean(v)) for m, v in marker_vals.items()}
    markers = sorted(means.keys())
    vals    = [means[m] for m in markers]

    # Achse erzeugen, falls nicht übergeben
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Balkendiagramm und Null-Linie
    bar_width = 0.8
    x = range(len(markers))
    ax.bar(markers, vals, width=bar_width)
    #ax.axhline(0, linewidth=1, color="black")
    ax.set_xlim(x[0] - bar_width/2, x[-1] + bar_width/2)

    # xticks
    ax.set_xticks(x)
    ax.set_xticklabels(markers, rotation=90, fontsize=5)
    
    dy = 0.015

    for label in ax.get_xticklabels():
        x, y = label.get_position()
        label.set_y(y + dy)

    ax.set_yticks([])
    # y-Ticks im ±tick_step-Raster um Null herum
    #y_min = min(min(vals), 0.0)
    #y_max = max(max(vals), 0.0)
    # float-Cast, dann math.floor/ceil
    #start = math.floor(y_min / tick_step) * tick_step
    #end   = math.ceil(y_max / tick_step) * tick_step
    #ticks = np.arange(start, end + tick_step/1000, tick_step)
    #ax.set_yticks(ticks)

    # sehr feine horizontale Gitterlinien
    #ax.yaxis.grid(True, which="major", linestyle="--", linewidth=1)

    # Beschriftungen und Titel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel,labelpad=30)
    ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_amplitude(
    amplitudes,
    ax=None,
    title=None,
    xlabel=None,
    ylabel=None,
    marked_indices=None,
    sampling_rate=100,
    marked_color='black',
    **plot_kwargs
):
    import matplotlib.ticker as ticker
    amplitudes = np.array(amplitudes)
    n_samples = len(amplitudes)
    duration = n_samples / sampling_rate

    # Zeitachse in Sekunden
    t = np.arange(n_samples) / sampling_rate

    if ax is None:
        fig, ax = plt.subplots()

    # Kurve zeichnen
    ax.plot(t, amplitudes, **plot_kwargs)

    # X‐Ticks alle 10 s
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%ds'))
    ax.set_xlim(0, duration)
    value_range = amplitudes.max() - amplitudes.min()
    ax.set_ylim(amplitudes.min() - value_range * 0.05, amplitudes.max() + value_range * 0.15)
    # Markierungen
    if marked_indices:
        # sortieren, damit das erste Element t1, das zweite t2 wird
        sorted_idx = sorted(marked_indices)
        ylim = ax.get_ylim()
        y_text = ylim[1] - (ylim[1] - ylim[0]) * 0.05  # 6% unter dem oberen Rand
        for idx in marked_indices:
            time_pt = idx / sampling_rate
            # gestrichelte Linie
            ax.axvline(x=time_pt, linestyle='--', color=marked_color, linewidth=0.8)
            # Label wählen
            if idx == sorted_idx[0]:
                lbl = f"$t_1$"
            elif len(sorted_idx) > 1 and idx == sorted_idx[1]:
                lbl = f"$t_2$"
            else:
                lbl = f"{time_pt:.2f}s"
            # unten mittig platzieren
            ax.text(
                time_pt - 5/sampling_rate ,
                y_text,
                lbl,
                rotation=0,
                verticalalignment='top',
                horizontalalignment='right',
                color=marked_color
            )

    # Achsenbeschriftungen
    ax.set_xlabel(xlabel or 'Time [s]')
    ax.set_ylabel(ylabel or 'Amplitude [mm]')
    if title:
        ax.set_title(title)

    return ax


def plot_bars(values,
              width=0.6,
              xlabel='Index',
              ylabel='Value',
              title='Bar Plot',
              xtick_step=5,
              annotate=True,
              highlights=None,
              special_ytick=None,
              ylim_padding=0.15):
    
    plt.style.use('ggplot')
    arr = np.asarray(values)
    n = len(arr)
    indices = np.arange(n)

    # Ensure valid xtick_step
    xtick_step = max(int(xtick_step), 1)

    fig, ax = plt.subplots()
    bars = ax.bar(indices, arr, width=width)

    # Apply highlights: first with //, second with \\ hatch
    if highlights is not None:
        hatches = ['//', 'xxx']  # two distinct hatch styles
        legend_handles = []
        for i, (label, idx) in enumerate(highlights.items()):
            if 0 <= idx < n:
                bar = bars[idx]
                hatch = hatches[i] if i < len(hatches) else hatches[0]
                bar.set_hatch(hatch)
                bar.set_edgecolor('black')
                bar.set_linewidth(1.2)
                legend_handles.append(
                    Patch(facecolor=bar.get_facecolor(),
                          edgecolor='black',
                          hatch=hatch,
                          label=f"{label} at index {idx + 1}")
                )
        if legend_handles:
            ax.legend(handles=legend_handles, title='')

    # Annotate bar heights
    if annotate:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center',
                        va='bottom')

    # X-axis ticks: first bar at index 1, then every xtick_step
    tick_positions = [0] + list(indices[(xtick_step - 1)::xtick_step])
    tick_labels = [str(pos + 1) for pos in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90)

    # Special y-tick and horizontal line
    if special_ytick is not None:
        # Add dashed horizontal line at the given y-value, matching bar color
        base_color = bars[0].get_facecolor() if n > 0 else 'black'
        ax.axhline(special_ytick, linestyle='--', linewidth=0.8, color=base_color)
        # Extend current y-ticks with the special value
        yticks = list(ax.get_yticks())
        yticks.append(special_ytick)
        ax.set_yticks(sorted(yticks))
        
    if ylim_padding is not None:
        y_min, y_max = arr.min(), arr.max()
        padding = ylim_padding * (y_max - y_min)
        ax.set_ylim(0, y_max + padding)

    # Labels, title, grid
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    