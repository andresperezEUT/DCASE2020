'''
localization_detection.py

Estimate DOAs and onsets/offsets from the given audio files.
Return results as a list of Events.
'''


from APRI.utils import *


def localization_detection_basic(stft, diff_th):

    M, K, N = stft.shape
    DOA = doa(stft) # Direction of arrival
    diff = diffuseness(stft) # Diffuseness
    diff_mask = diff <= diff_th

    # segment audio based on diffuseness mask
    source_activity = np.empty(N)
    for n in range(N):
        source_activity[n] = np.any(diff_mask[:,n]) # change here discriminative function

    # compute statistics of relevant DOAs
    active_frames = np.argwhere(source_activity>0).squeeze()
    num_active_frames = active_frames.size
    estimated_doa_per_frame = np.empty((num_active_frames,2))

    for af_idx, af in enumerate(active_frames):
        active_bins = diff_mask[:,af]
        doas_active_bins = DOA[:,active_bins,af]
        for a in range(2): # angle
            estimated_doa_per_frame[af_idx,a] = circmedian(doas_active_bins[a])

    # segmentate active bins into "events"
    frame_changes = np.argwhere(active_frames[1:] - active_frames[:-1] != 1).flatten()
    frame_changes = np.insert(frame_changes, 0, -1)
    event_list = []
    for idx in range(len(frame_changes)-1):
        start_frame_idx = frame_changes[idx]+1
        end_frame_idx = frame_changes[idx+1]
        frames = active_frames[start_frame_idx:end_frame_idx+1]
        azis = estimated_doa_per_frame[start_frame_idx:end_frame_idx + 1, 0]
        eles = estimated_doa_per_frame[start_frame_idx:end_frame_idx + 1, 1]
        event_list.append(Event(-1, -1, frames, azis, eles))

    return event_list

