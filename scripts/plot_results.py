import json
from pathlib import Path

from dynamic_components.config import settings, current_subject
from dynamic_components.DycaReconstructor import ReconstructionResult
from dynamic_components.Subject import Subject
from dynamic_components.Dyca import Dyca
from dynamic_components.DycaResult import DycaResult
import importlib
from dynamic_components import utils
importlib.reload(utils)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def create_plot(rec_dict, plot_config,figsize=(8, 6), sharex=True):
    #Select from rec_dict
    Q_det = rec_dict['Q_det']
    mode_vector = rec_dict['modes']
    amplitude = rec_dict['amplitudes'].squeeze()
    amplitude = amplitude[0:plot_config.get('amplitude_scope', 500)]
    selected_amplitude = rec_dict['selected_amplitudes']

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1], wspace=0.3)
    
    gs_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.05)
    ax0 = fig.add_subplot(gs_left[0], projection='3d')
    ax1 = fig.add_subplot(gs_left[1], projection='3d')
    
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.3)
    ax2 = fig.add_subplot(gs_right[0])
    ax3 = fig.add_subplot(gs_right[1])
    
    #Parameter für freeze_frames
    frame_indice_1 = np.argmax(amplitude)
    frame_indice_2 = np.argmin(amplitude)
    elev=plot_config.get('elev', 30)
    azim=plot_config.get('azim', -60)

    #-------------Plots-------------
    #Plot 1: Freeze Frame
    utils.plot_freeze_frame(
        Q_det,
        frame_index=frame_indice_1,
        title_prefix="Frame",
        #plot_title=f"Freeze Frame number {selected_amplitude}",
        draw_lines=True,
        elev=elev,
        azim=azim,
        ax=ax0
    )
    ax0.set_title('Freeze Frames')
    
    #Plot 2: Freeze Frame
    utils.plot_freeze_frame(
        Q_det,
        frame_index=frame_indice_2,
        title_prefix="Frame",
        #plot_title=f"Freeze Frame number {selected_amplitude}",
        draw_lines=True,
        elev=elev,
        azim=azim,
        ax=ax1
    )
    ax1.set_title('Freeze Frames')

    #Plot 3: Amplitude
    utils.plot_amplitude(
        amplitude,
        marked_indices=[frame_indice_1, frame_indice_2],
        ax=ax2
    )
    ax2.set_title('Amplitude')
    
    
    #Plot 4: Mode
    channel_names = Q_det.columns.drop('Frame').tolist()
    utils.plot_modes(
        mode_vector,
        channel_names,
        ax=ax3
    )
    ax3.set_title('Modi')

    fig.suptitle(f"Reconstruction: {None}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def run_all(plot_config, subject, iteration):
    for amp in range(len(iteration.dyca_output['amplitudes_svd'])):
        if amp != 0:
            continue
        rec_dict = iteration.do_reconstruction(subject, selected_amplitudes=amp)
        # Retransforming modes to original space
        pca_model = subject.pipeline.steps[-1][1]._pca_model
        mode_vector = rec_dict['modes'].reshape(-1)
        mode_vector_origin = mode_vector @ pca_model.components_
        rec_dict['modes'] = mode_vector_origin
        
        fig = create_plot(
            rec_dict=rec_dict,
            plot_config=plot_config,
            figsize=plot_config.get('figsize', (8, 6)),
            sharex=plot_config.get('sharex', True),
            )

        if plot_config.get('save_flag', False):
            outfile = plot_config['save_path'] / f"{plot_config['dataset_id']}_{amp}_plot.png"
            fig.savefig(outfile, dpi=300)
            print(f"Plot gespeichert: {outfile}")
        else:
            plt.show()

        plt.close(fig)
            
BASE_DIR        = Path(__file__).resolve().parents[1] 
PROCESSED_DIR   = BASE_DIR / 'data' / 'processed'
RESULTS_DIR     = BASE_DIR / 'results' 

if __name__ == '__main__':
    
    plot_config = {
    'figsize': (8, 6),
    'sharex': True,
    'elev': 30,
    'azim': -60,
    'amplitude_scope': 500,  # Number of samples to show in amplitude plot
    'save_flag': False,
    'save_path': Path('results/plots'),
    'dataset_id': 'UNKNOWN',
}
    
    filename = 'Sub8_Kinematics_T3'
    load_path = PROCESSED_DIR / 'preprocessed_data' / (Path(filename).stem + '.pkl')
    subject = Subject.load_subject(load_path)
    dyca = Dyca(subject)
    plot_config['dataset_id'] = subject.dataset_id
    dyca_result = dyca.run(m=9, n=11, custom_key='custom')
    dyca_result.dataset_id = filename
    iteration=dyca_result.get_iteration_by_index(1)
    
    run_all(plot_config, subject, iteration)
    
