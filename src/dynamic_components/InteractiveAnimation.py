import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dynamic_components.config import settings

MARKER_LOC = settings.marker_loc

class PlotlyMotionCaptureAnimation:
    def __init__(self, df: pd.DataFrame, title="Motion Capture", draw_lines: bool = True,  marker_colors: list[str] | None = None):
        self.df = df
        self.title = title
        self.draw_lines = draw_lines
        self.frames = []
        self.marker_colors = marker_colors or ['blue'] * (df.shape[1]//3)
        
        self._compute_axes_limits()
        print(f"x_range: {self.x_range}")
        print(f"y_range: {self.y_range}")
        print(f"z_range: {self.z_range}")
        self._prepare_frames()
        self._create_figure()

    def _compute_axes_limits(self):
        
        xs = self.df.iloc[:, 1::3].values.flatten()
        ys = self.df.iloc[:, 2::3].values.flatten()
        zs = self.df.iloc[:, 3::3].values.flatten()
        
        def with_margin(arr):
            mn, mx = np.min(arr), np.max(arr)
            m = (mx-mn)*0.05
            return mn - m, mx + m
        self.x_range = with_margin(xs)
        self.y_range = with_margin(ys)
        self.z_range = with_margin(zs)

    def _prepare_frames(self):
        for _, row in self.df.iterrows():
            xs = row.iloc[1::3].values
            ys = row.iloc[2::3].values
            zs = row.iloc[3::3].values
            
            color_map = {
                'blue':  '#0000FF',
                'green': '#00FF00',
                'red':   '#FF0000'
            }
            hex_colors = [color_map[c] for c in self.marker_colors]

            data = [go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers', marker=dict(size=8, color=hex_colors),
                showlegend=False
            )]
            if self.draw_lines:
                for conns in MARKER_LOC.values():
                    for i, j in conns:
                        data.append(go.Scatter3d(
                            x=[xs[i], xs[j]],
                            y=[ys[i], ys[j]],
                            z=[zs[i], zs[j]],
                            mode='lines',
                            line=dict(width=2, color='black'),
                            showlegend=False
                        ))
            self.frames.append(go.Frame(data=data, name=str(int(row['Frame']))))

    def _create_figure(self):
        init_data = self.frames[0].data if self.frames else []
        
        ranges = [self.x_range[1] - self.x_range[0],
                        self.y_range[1] - self.y_range[0],
                        self.z_range[1] - self.z_range[0]]
        min_range = min(ranges)
        scaled_ranges = [r / min_range for r in ranges]

        default_camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))

        self.fig = go.Figure(data=init_data, frames=self.frames)
        self.fig.update_layout(
            title=self.title,
            scene=dict(
                xaxis=dict(range=list(self.x_range), autorange=False),
                yaxis=dict(range=list(self.y_range), autorange=False),
                zaxis=dict(range=list(self.z_range), autorange=False),
                aspectmode='manual',
                #aspectmode='cube',
                #aspectmode='data',
                aspectratio=dict(x=scaled_ranges[0], y=scaled_ranges[1], z=scaled_ranges[2]),
                camera=default_camera
            ),
            updatemenus=[dict(
                type='buttons', showactive=False,
                y=1, x=1.15, xanchor='right', yanchor='top',
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=10, redraw=True),
                                          fromcurrent=True, transition=dict(duration=0))]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode='immediate', transition=dict(duration=0))])
                ]
            )],
            sliders=[dict(
                active=0, pad=dict(t=50),
                steps=[dict(label=fr.name,
                            method='animate',
                            args=[[fr.name], dict(frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0))])
                       for fr in self.frames]
            )]
        )

        cam = self.fig.layout.scene.camera.eye
        az = np.degrees(np.arctan2(cam.y, cam.x))
        el = np.degrees(np.arcsin(cam.z / np.sqrt(cam.x**2 + cam.y**2 + cam.z**2)))
        zoom = 1 / np.linalg.norm([cam.x, cam.y, cam.z])
        self.fig.add_annotation(
            showarrow=False,
            text=f"Az: {az:.1f}° | El: {el:.1f}° | Zoom: {zoom:.2f}",
            xref='paper', yref='paper',
            x=0.95, y=0.05,
            bgcolor='rgba(255,255,255,0.7)',
            font=dict(size=12)
        )

    def save_html(self, filename: str = 'motion_capture_animation.html', embed_live_camera: bool = True):
        html = self.fig.to_html(full_html=False)

        if embed_live_camera:
            # Beispiel-Hook: horcht auf plotly_relayout und passt Annotation an
            js = """
                <script>
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                gd.on('plotly_relayout', function(eventdata) {
                    if (eventdata['scene.camera']) {
                        var c = eventdata['scene.camera'].eye;
                        var az = Math.atan2(c.y, c.x)*180/Math.PI;
                        var el = Math.asin(c.z/Math.sqrt(c.x*c.x + c.y*c.y + c.z*c.z))*180/Math.PI;
                        var zoom = 1/Math.sqrt(c.x*c.x + c.y*c.y + c.z*c.z);
                        Plotly.relayout(gd, {'annotations[0].text': 
                            'Az: '+az.toFixed(1)+'° | El: '+el.toFixed(1)+'° | Zoom: '+zoom.toFixed(2)
                        });
                    }
                });
                </script>
                """
            html += js

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"HTML-Datei gespeichert unter: {filename}")