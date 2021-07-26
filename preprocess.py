import mne
import pickle

def preprocess(input_file_name, output_file_name, baseline_start=0, experiment_start=4, experiment_end=None):
    raw = mne.io.read_raw_edf(input_file_name, preload=True)
    eeg = raw.copy().pick_types(meg=False, eeg=True, exclude=[])
    eeg_indices = mne.pick_channels_regexp(eeg.ch_names, '^EEG')
    eeg.info = mne.pick_info(eeg.info, eeg_indices)

    map_1020_to_1010 = {
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8'
    }

    eeg.rename_channels(lambda s: s.replace("EEG ", ""))
    mne.rename_channels(eeg.info, map_1020_to_1010)
    geneva_channel_order = ['Fp1', 'F3', 'F7', 'C3', 'T7', 'P3', 'P7', 'O1', 'Pz', 'Fp2', 'Fz', 'F4', 'F8', 'Cz', 'C4', 'T8', 'P4', 'P8', 'O2']
    eeg.reorder_channels(geneva_channel_order)

    montage = mne.channels.make_standard_montage("standard_1020")
    eeg.set_montage(montage, match_case=False)

    # DEAP Preprocessing

    # The data was downsampled to 128Hz.
    eeg.resample(128, npad="auto")

    # EOG artefacts were removed as in [1].

    # A bandpass frequency filter from 4.0-45.0Hz was applied.
    eeg.filter(4.0, 45.0)

    # The data was averaged to the common reference.
    eeg.set_eeg_reference("average")

    baseline_selection = eeg.copy().crop(tmin=baseline_start, tmax=baseline_start + 3)
    eeg_selection = eeg.copy().crop(tmin=experiment_start, tmax=experiment_end)

    eeg_combined = mne.concatenate_raws([baseline_selection, eeg_selection])

    output = {
        'data': eeg_combined.get_data()
    }

    with open(output_file_name, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
