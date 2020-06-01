# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.

# set user
import os
import warnings
import matplotlib.pyplot as plt
environ_user = os.environ.get('USER')
user = None
if environ_user == 'andres.perez':
    user = 'PANS'
elif environ_user == 'ribanez': #
    user = 'FAIK'
else:
    warnings.warn('user not known', UserWarning)



def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # # INPUT PATH
        # dataset_dir='/scratch/asignal/sharath/DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders

        # OUTPUT PATH
        # feat_label_dir='/scratch/asignal/sharath/DCASE2020_SELD_dataset/feat_label/',  # Directory to dump extracted features and labels
        model_dir='models/',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='results/',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        num_classes = 14,
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        # DNN MODEL PARAMETERS
        label_sequence_length=60,        # Feature sequence length
        batch_size=256,              # Batch size
        dropout_rate=0,             # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        rnn_size=[128, 128],        # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],             # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 1000.],     # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=50,               # Train for maximum epochs
        epochs_per_fit=5,           # Number of epochs per fit
        doa_objective='masked_mse',     # supports: mse, masked_mse. mse- original seld approach; masked_mse - dcase 2020 approach
        
        #METRIC PARAMETERS
        lad_doa_thresh=20,

        # APRI PARAMETERS

        results_dir = 'results',

        window = 'boxcar',
        window_size = 2400,
        window_overlap = 0,
        nfft = 2400,
        D = 10, # decimation

    )
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    params['unique_classes'] = {
            'alarm': 0,
            'baby': 1,
            'crash': 2,
            'dog': 3,
            'engine': 4,
            'female_scream': 5,
            'female_speech': 6,
            'fire': 7,
            'footsteps': 8,
            'knock': 9,
            'male_scream': 10,
            'male_speech': 11,
            'phone': 12,
            'piano': 13
        }

    # INPUT PATH
    # dataset_dir: Base folder containing the foa/mic and metadata folders
    if user == 'PANS':
        params['dataset_dir'] = '/Volumes/Dinge/datasets/DCASE2020_TASK3'
    elif user == 'FAIK':
        params['dataset_dir'] = '/home/ribanez/movidas/dcase20/dcase20_dataset'
    else:
        warnings.warn('user not known', UserWarning)

    # OUTPUT PATH
    # Directory to dump extracted features and labels
    if user == 'PANS':
        params['feat_label_dir'] ='/Volumes/Dinge/datasets/DCASE2020_TASK3/feat_label/'
    elif user == 'FAIK':
        params['feat_label_dir'] = '/home/ribanez/movidas/dcase20/'
    else:
        warnings.warn('user not known', UserWarning)

    # MATPLOTLIB BACKEND
    if user == 'PANS':
        plt.switch_backend('MacOSX')
    elif user == 'FAIK':
        plt.switch_backend('TKagg')




    # ########### User defined parameters ##############

    if argv == 'oracle_beam':
        # localization_detection
        params['ld_method'] = 'ld_oracle'
        # beamforming
        params['beamforming_mode'] = 'beam'
        # classification
        params['class_method'] = 'event_class_prediction'
        params['class_method_args'] = ['event_class_xgb']
        params['preset_descriptor'] = 'loc:metadata; beam:beam; cls:xgb'

    if argv == 'alpha_v1':
        # localization_detection
        params['ld_method'] = 'ld_basic'
        params['ld_method_args'] = [0.3] # [diff_th]
        # beamforming
        params['beamforming_mode'] = 'beam'
        # classification
        params['class_method'] = 'event_class_prediction_random'
        params['class_method_args'] = []
        # postprocessing
        params['event_filter_activation']= False
        params['event_filter_method'] = ''
        params['event_filter_method_args']=[] #[frames threshold, frames_threshold_fp_8]

        params['preset_descriptor'] = 'loc:basic; beam:beam; cls:random'

    if argv == 'alpha_v2':
        # localization_detection
        params['ld_method'] = 'ld_basic'
        params['ld_method_args'] = [0.3] # [diff_th]
        # beamforming
        params['beamforming_mode'] = 'beam'
        # classification
        params['class_method'] = 'event_class_prediction'
        params['class_method_args'] = ['event_class_rf']
        # postprocessing
        params['event_filter_activation']= False
        params['event_filter_method'] = ''
        params['event_filter_method_args']=[] #[frames_threshold_fp_8]

        params['preset_descriptor'] = 'loc:basic; beam:beam; cls:random_forest'

    if argv == 'mi_primerito_dia':
        # localization_detection
        params['ld_method'] = 'ld_basic_dereverb_filter'
        params['ld_method_args'] = [0.3, 5, 4]  # [diff_th, L, event_minimum_length]
        # beamforming
        params['beamforming_mode'] = 'beam'
        # classification
        params['class_method'] = 'event_class_prediction'
        params['class_method_args'] = ['event_class_rf']
        # postprocessing
        params['event_filter_activation']= False
        params['event_filter_method'] = ''
        params['event_filter_method_args']=[] #[frames_threshold_fp_8]

        params['preset_descriptor'] = 'loc:dereverb_filter; beam:beam; cls:random_forest'

    if argv == 'mi_primerito_dia_postfilter':
        # localization_detection
        params['ld_method'] = 'ld_basic_dereverb_filter'
        params['ld_method_args'] = [0.3, 5, 4]  # [diff_th, L, event_minimum_length]
        # beamforming
        params['beamforming_mode'] = 'beam'
        # classification
        params['class_method'] = 'event_class_prediction'
        params['class_method_args'] = ['event_class_rf']
        # postprocessing
        params['event_filter_activation']= True
        params['event_filter_method'] = 'event_filter_v1'
        params['event_filter_method_args']=[10] #[frames_threshold_fp_8]

        params['preset_descriptor'] = 'loc:dereverb_filter; beam:beam; cls:random_forest; postfilter:v1'

    if argv == 'mi_primerito_dia_xgb':
        # localization_detection
        params['ld_method'] = 'ld_basic_dereverb_filter'
        params['ld_method_args'] = [0.3, 5, 4]  # [diff_th, L, event_minimum_length]
        # beamforming
        params['beamforming_mode'] = 'beam'
        # classification
        params['class_method'] = 'event_class_prediction'
        params['class_method_args'] = ['event_class_xgb']
        # postprocessing
        params['event_filter_activation']= True
        params['event_filter_method'] = 'event_filter_v1'
        params['event_filter_method_args']=[10] #[frames_threshold_fp_8]

        params['preset_descriptor'] = 'loc:dereverb_filter; beam:beam; cls:xgb'

    if argv == 'new_features_xgb':
        # localization_detection
        params['ld_method'] = 'ld_basic_dereverb_filter'
        params['ld_method_args'] = [0.3, 5, 4]  # [diff_th, L, event_minimum_length]
        # beamforming
        params['beamforming_mode'] = 'beam'
        # classification
        params['class_method'] = 'event_class_prediction'
        params['class_method_args'] = ['event_class_xgb_2']
        # postprocessing
        params['event_filter_activation'] = True
        params['event_filter_method'] = 'event_filter_v2'
        params['event_filter_method_args'] = [10]  # [frames_threshold_fp_3]

        params['preset_descriptor'] = 'loc:dereverb_filter; beam:beam; cls:xgb2; postfilter'

    else:
        print('ERROR: unknown argument {}'.format(argv))
        # exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
