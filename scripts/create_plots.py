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


def create_freeze_frames_plot(rec_dict, plot_config, figsize=(8, 6)):
    plt.style.use('ggplot')
    """
    Erzeugt eine Figur mit zwei 3D-Subplots nebeneinander und zeichnet jeweils einen Freeze Frame.
    """
    Q_det = rec_dict['Q_det']
    amplitude = rec_dict['amplitudes'].squeeze()
    amplitude = amplitude[: plot_config.get('amplitude_scope', 501)]
    # Indizes für max und min Amplitude
    frame_max = np.argmax(amplitude)
    try:
        frame_min = np.argmin(amplitude[frame_max:frame_max+120]) + frame_max
    except ValueError:
        frame_max = np.argmax(amplitude[:300])
        frame_min = np.argmin(amplitude[frame_max:frame_max+120]) + frame_max
    elev = plot_config.get('elev', 30)
    azim = plot_config.get('azim', -60)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=-0.23)
    ax0 = fig.add_subplot(gs[0], projection='3d')
    ax1 = fig.add_subplot(gs[1], projection='3d')

    # Plot Freeze Frame (max amplitude)
    utils.plot_freeze_frame_v2(
        Q_det,
        frame_index=frame_max,
        title_prefix="Frame",
        draw_lines=True,
        elev=elev,
        azim=azim,
        ax=ax0,
        trail_start=None,
    )
    amp = rec_dict['selected_amplitudes'] + 1
    ax0.set_title(f"$dc_{{{amp}}}(t_1)$", y = -0.1)


    # Plot Freeze Frame (min amplitude)
    utils.plot_freeze_frame_v2(
        Q_det,
        frame_index=frame_min+500,
        title_prefix="Frame",
        draw_lines=True,
        elev=elev,
        azim=azim,
        ax=ax1,
        trail_start=None,
    )

    ax1.set_title(f"$dc_{{{amp}}}(t_2)$", y=-0.1)

    fig.suptitle('')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_amplitude_modes_plot(rec_dict, plot_config, figsize=(8, 6), sharex=True):
    """
    Erzeugt eine Figur mit zwei 2D-Subplots untereinander: Amplitude und Modi.
    """
    mode_vector = rec_dict['modes']
    amplitude = rec_dict['amplitudes'].squeeze()
    #Reskalieren
    norm_mode = np.linalg.norm(mode_vector)
    try:
        mode_vector = mode_vector / norm_mode
        amplitude = amplitude * norm_mode
    except Exception as e:
        print(f"Fehler beim Reskalieren {e}")

    amplitude = amplitude[: plot_config.get('amplitude_scope', 501)]
    # Indizes für Markierungen
    frame_max = np.argmax(amplitude)
    try:
        frame_min = np.argmin(amplitude[frame_max:frame_max+120]) + frame_max
    except ValueError:
        frame_max = np.argmax(amplitude[:300])
        frame_min = np.argmin(amplitude[frame_max:frame_max+120]) + frame_max
    channel_names = rec_dict['Q_det'].columns.drop('Frame').tolist()
    fig, (ax2, ax3) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=figsize,
        sharex=sharex,
        gridspec_kw={'hspace': 0.3}
    )

    # Amplitude-Plot
    utils.plot_amplitude(
        amplitude,
        marked_indices=[frame_max, frame_min],
        ax=ax2
    )
    ax2.set_title('')

    # Modi-Plot
    utils.plot_modes(
        mode_vector,
        channel_names,
        ax=ax3
    )
    ax3.set_title('')

    fig.tight_layout()
    return fig


def run_all(plot_config, subject, iteration):
    for amp in range(len(iteration.dyca_output['amplitudes_svd'])):
        # if amp != 0:
        #     continue
        rec_dict = iteration.do_reconstruction(subject, selected_amplitudes=amp)
        #Berechne ungewichtete Varianz
        singular_values = iteration.dyca_output['singular_values']
        σ_i = singular_values[amp]                             # aktueller Singulärwert
        raw_variance = σ_i**2                                  # ungewichtete Varianz
        variance_ratio = raw_variance / np.sum(singular_values**2)
        # print("amplitude:", amp, "raw_variance:", raw_variance, "variance_ratio:", variance_ratio)


        # ——— Gesamtvarianz der Original-Bewegung ———
        X = subject.undo_processing(upto='pos_center')
        X = X.drop(columns=['Frame'], inplace=True, errors='ignore')  # Entferne 'Frame' Spalte, falls vorhanden
        try:
            X = X.to_numpy()
        except AttributeError:
            print("Cannot convert DataFrame to numpy array, check the data structure")
            pass
        Xc = X - np.mean(X, axis=0, keepdims=True)
        total_variance = np.sum(np.var(Xc, axis=0, ddof=0))
        # ————————————————————————————————

        # ——— Varianz des ausgewählten Modus ———
        sv = iteration.dyca_output['singular_values']
        σ_i = sv[amp]
        var_i = σ_i**2
        frac_i = var_i / total_variance

        # Ins rec_dict speichern
        rec_dict['mode_variance'] = var_i
        rec_dict['mode_variance_ratio'] = frac_i

        print(f"[Amp {amp}] σ²={var_i:.3f} → {frac_i*100:.1f}% der Gesamtvarianz")
        # ————————————————————————————————
        
        
        # Retransforming modes to original space
        pca_model = subject.pipeline.steps[-1][1]._pca_model
        mode_vector = rec_dict['modes'].reshape(-1)
        mode_vector_origin = mode_vector @ pca_model.components_
        rec_dict['modes'] = mode_vector_origin

        # Erzeuge und zeige/speichere die Freeze Frame Subplots
        # fig1 = create_freeze_frames_plot(
        #     rec_dict=rec_dict,
        #     plot_config=plot_config,
        #     figsize=plot_config.get('figsize', (8, 6))
        # )

        # # Erzeuge und zeige/speichere den Amplitude/Modi Subplot
        # fig2 = create_amplitude_modes_plot(
        #     rec_dict=rec_dict,
        #     plot_config=plot_config,
        #     figsize=plot_config.get('figsize', (8, 6)),
        #     sharex=False
        # )
        # if plot_config.get('save_flag', False):
        #     save_path = plot_config['save_path']
        #     dataset_id = plot_config.get('dataset_id', 'UNKNOWN')
        #     out1 = save_path / f"{dataset_id}_{amp}_freeze.png"
        #     out2 = save_path / f"{dataset_id}_{amp}_amp_modes.png"
        #     fig1.savefig(out1, dpi=300)
        #     fig2.savefig(out2, dpi=300)
        #     print(f"Plots gespeichert: {out1}, {out2}")
        # else:
        #     plt.show(fig1)
        #     plt.show(fig2)

        # plt.close(fig1)
        # plt.close(fig2)


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
RESULTS_DIR = BASE_DIR / 'results'

if __name__ == '__main__':
    plot_config = {
        'figsize': (8, 6),
        'sharex': True,
        'elev': 30,
        'azim': -60,
        'amplitude_scope': 501,
        'save_flag': False,
        'save_path': RESULTS_DIR / 'plots',
        'dataset_id': 'UNKNOWN',
    }

    filename = 'Sub8_Kinematics_T3'
    load_path = PROCESSED_DIR / 'preprocessed_data' / (Path(filename).stem + '.pkl')
    subject = Subject.load_subject(load_path)
    dyca = Dyca(subject)
    plot_config['dataset_id'] = subject.dataset_id
    dyca_result = dyca.run(m=9, n=11, custom_key='custom')
    dyca_result.dataset_id = filename
    iteration = dyca_result.get_iteration_by_index(1)

    run_all(plot_config, subject, iteration)
