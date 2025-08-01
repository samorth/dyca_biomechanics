import logging
import numpy as np
import pandas as pd
from pathlib import Path

from dynamic_components.DycaResult import DycaResult
from dynamic_components import utils
from dynamic_components.Subject import Subject
from dynamic_components.config import setup_logging, current_subject
from dynamic_components.Animation import MotionCaptureAnimation
from dynamic_components.DycaReconstructor import ReconstructionResult
from dynamic_components.Reconstructor import Reconstructor
from dynamic_components.Dyca import Dyca
#from dynamic_components.InteractiveAnimation import PlotlyMotionCaptureAnimation

logger = logging.getLogger(__name__)

def animate(Q_det, anim_config):
    # Basis-Parameter
    time_horizon = anim_config.get('time_horizon', 1000)
    interval     = anim_config.get('interval', 50)
    xlims        = tuple(anim_config.get('xlims', (-300, 300)))
    ylims        = tuple(anim_config.get('ylims', (-450, 450)))
    zlims        = tuple(anim_config.get('zlims', (-1100, 500)))
    draw_lines   = anim_config.get('draw_lines', True)
    save_flag    = anim_config.get('save_flag', False)
    save_dir     = Path(anim_config.get('save_path', 'results/animations'))

    # DYCA-spezifische Meta-Infos
    subject_id = anim_config.get('subject', '')
    m          = anim_config.get('dyca_m', anim_config.get('m', None))
    n          = anim_config.get('dyca_n', None)
    index      = anim_config.get('index', None)

    # Title-Template anpassen (falls custom übergeben)
    title_tpl = anim_config.get(
        'title',
        'Motion Capture Animation'
    )
    title = title_tpl.format(
        subject=subject_id,
        m=m,
        n=n,
        index=index
    )

    # Daten auf Zeit-Horizont beschränken
    Q_slice = Q_det[:time_horizon]

    # Animation erstellen
    anim = MotionCaptureAnimation(
        Q_slice,
        interval   = interval,
        title      = title,
        draw_lines = draw_lines,
        xlims      = xlims,
        ylims      = ylims,
        zlims      = zlims,
    )
    
    # Abspeichern (wenn gewünscht)
    if save_flag:
        from matplotlib.animation import FFMpegWriter
        save_dir.mkdir(parents=True, exist_ok=True)
        # Dateiname mit allen relevanten Infos
        fname = f"{subject_id}_amp_{index:02d}"
        if n is not None:
            fname += f"_n_{n:02d}"
        filename = save_dir / f"{fname}.mp4"
        
        fps = int(1000 / interval)  # or whatever fps you like
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='You'), bitrate=1800)

        # pass the writer object, not a string
        anim.save_animation(str(filename), writer=writer)
        logger.info(f"Animation gespeichert unter: {filename}")

    return anim

        
def run_all(anim_config, subject, iteration):
    for amp in range(len(iteration.dyca_output['amplitudes_svd'])):
        rec_dict = iteration.do_reconstruction(subject, selected_amplitudes=amp)
        Q_det = rec_dict['Q_det']
        cfg = {
            **anim_config,
            'subject': subject.dataset_id,
            'index': amp+1,
            'dyca_m': iteration.m,
            'dyca_n:': iteration.n,
            'title': f"DYCA Animation - {subject.dataset_id} - amp={amp+1}, m={iteration.m}, n={iteration.n}"
            }
        animate(Q_det, cfg)

        
BASE_DIR        = Path(__file__).resolve().parents[1] 
PROCESSED_DIR   = BASE_DIR / 'data' / 'processed'
RESULTS_DIR     = BASE_DIR / 'results' 


if __name__ == '__main__':
    anim_config = {
    
    "time_horizon": 1000,         
    "interval": 50,                

    "xlims": (-300, 300),
    "ylims": (-450, 450),
    "zlims": (-1100, 500),
    "draw_lines": True,

    "save_flag": True,
    "save_path": "results/animations/svd",
}
    
    filename = 'Sub8_Kinematics_T3'
    load_path = PROCESSED_DIR / 'preprocessed_data' / (Path(filename).stem + '.pkl')
    subject = Subject.load_subject(load_path)
    dyca = Dyca(subject)
    dyca_result = dyca.run(m=9, n=11, custom_key='custom')
    dyca_result.dataset_id = filename
    iteration=dyca_result.get_iteration_by_index(1)
    
    run_all(anim_config, subject, iteration)