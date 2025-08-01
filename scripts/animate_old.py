import logging
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from dynamic_components.DycaResult import DycaResult
from dynamic_components import utils
from dynamic_components.Subject import Subject
from dynamic_components.config import setup_logging, current_subject
from dynamic_components.Animation import MotionCaptureAnimation
from dynamic_components.DycaReconstructor import ReconstructionResult
#from dynamic_components.InteractiveAnimation import PlotlyMotionCaptureAnimation

logger = logging.getLogger(__name__)

def animate(Q_det, anim_config):
    time_horizon = anim_config.get('time_horizon', 1000)
    interval     = anim_config.get('interval', 50)
    title        = anim_config.get('title', 'Motion Capture Animation')
    xlims        = tuple(anim_config.get('xlims', (-300, 300)))
    ylims        = tuple(anim_config.get('ylims', (-450, 450)))
    zlims        = tuple(anim_config.get('zlims', (-1100, 500)))
    draw_lines   = anim_config.get('draw_lines', True)
    save_flag    = anim_config.get('save_flag', False)
    save_dir     = Path(anim_config.get('save_path', 'results/animations'))
    m            = anim_config.get('m', None)
    index        = anim_config.get('index', None)
    kind         = anim_config.get('kind', '')

    Q_slice = Q_det[:time_horizon]

    title = title.format(kind=kind, index=index, m=m)

    anim = MotionCaptureAnimation(
        Q_slice,
        interval   = interval,
        title      = title,
        draw_lines = draw_lines,
        xlims      = xlims,
        ylims      = ylims,
        zlims      = zlims,
    )
    
    if save_flag:
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"{kind}_amp_{index:02d}_of_{m:02d}.mp4"
        anim.save_animation(filename=filename, writer='ffmpeg')
        
    if anim_config.get('start_flag', True):
        anim.start_animation()
        
    return anim

        
def main(anim_config):
    load_path = PROCESSED_DIR / 'reconstruction_objects' / 'Sub8_Kinematics_T3_reconstructions.pkl'
    reconstructions = ReconstructionResult.load(load_path)
    reconstructions = reconstructions.select(
        kinds=anim_config['kinds'],
        indices=anim_config['indices']
    )
    
    for kind, Q_det_list in reconstructions.reconstructions.items():
        for idx, Q_det in enumerate(Q_det_list):
            cfg = {
                **anim_config,
                'kind': kind,
                'index': idx,
                'm': len(Q_det_list),
            }
        for idx, Q_det in enumerate(Q_det_list):
            is_last = (idx == len(Q_det_list) - 1)
            cfg = {
                 **anim_config,
                'kind':  kind,
                'index': idx,
                'm':     len(Q_det_list),
                'start_flag': is_last,
            }
            animate(Q_det, cfg)
        
    
    
BASE_DIR        = Path(__file__).resolve().parents[1] 
PROCESSED_DIR   = BASE_DIR / 'data' / 'processed'
RESULTS_DIR     = BASE_DIR / 'results' 


if __name__ == '__main__':
    anim_config = {
    
    "time_horizon": 1000,         
    "interval": 50,                

    "title": "Subject {kind} amplitude {index} of {m}",  
    "xlims": (-300, 300),
    "ylims": (-450, 450),
    "zlims": (-1100, 500),
    "draw_lines": True,

    "save_flag": False,
    "save_path": "results/animations",

    "kinds": ["position",
            'velocity', 
            #"combined"
            ],  
    "indices": {
        "position": [0],
        "velocity": [0],
        #"combined": None
    }
}
    
    main(anim_config)
    