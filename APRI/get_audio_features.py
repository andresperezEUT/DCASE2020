"""
get_audio_features.py

This script contains methods for audio features calculation os an audio file.
The audio features are calculated using essentia library and the default parameters are:
- samplerate: 24000
- framesize: 2048
- hopsize: 1024

compute_audio_features() is used in both train and predict circuits and takes as inputs:
- audio: event to be classified in audio file .wav
- options: parameters of feature extraction
and outputs:
- audio_features: numpy array with audio feature values
- column_labels: audio feature descriptors

"""

# Dependencies
from essentia.standard import *
import numpy as np



# Auxiliar
def is_silent_threshold(frame, silence_threshold_dB):
    p = essentia.instantPower( frame )
    silence_threshold = pow(10.0, (silence_threshold_dB / 10.0))
    if p < silence_threshold:
       return 1.0
    else:
       return 0.0
def spectralContrastPCA(scPool):
    scCoeffs = scPool['lowlevel.sccoeffs']
    scValleys = scPool['lowlevel.scvalleys']
    frames = len(scCoeffs)
    coeffs = len(scCoeffs[0])
    merged = np.zeros(2*coeffs, dtype='f4')
    for i in range(frames):
        k = 0
        for j in range(coeffs):
            merged[k] = scCoeffs[i][j]
            merged[k+1] = scValleys[i][j]
            k+= 2
        scPool.add('contrast', merged)
    pca = PCA(namespaceIn = 'contrast', namespaceOut = 'contrast')(scPool)
    pca = np.array(pca['contrast'])
    return pca
def normalize(hpcp):
    m = max(hpcp)
    for i in range(len(hpcp)):
        hpcp[i] = hpcp[i] / m
    return hpcp
def compute_statistics(array):
    array_mean = []
    array_var = []
    for i in range(array.shape[1]):
        aux=np.mean(array[:, i])
        array_mean.append(aux)
        aux=np.var(array[:,i])
        array_var.append(aux)
    array_mean = np.array(array_mean)
    array_var = np.array(array_var)
    return [array_mean, array_var]

# Calculate audio features
def compute_lowlevel(audio, options):
    namespace = 'lowlevel'
    pool = essentia.Pool()
    pool2 = essentia.Pool()
    # analysis parameters
    sampleRate = options['sampleRate']
    frameSize  = options['frameSize']
    hopSize    = options['hopSize']

    # temporal descriptors

    zerocrossingrate = ZeroCrossingRate()

    # frame algorithms
    frames = FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    window = Windowing(size = frameSize, zeroPadding = 0)
    spectrum = Spectrum(size = frameSize)

    # spectral algorithms
    lpc = LPC(order = 10, type = 'warped', sampleRate = sampleRate)
    barkbands = BarkBands(sampleRate=sampleRate)
    centralmoments = CentralMoments()
    crest = Crest()
    centroid = Centroid()
    decrease = Decrease()
    spectral_contrast = SpectralContrast(frameSize=frameSize, sampleRate=sampleRate, numberBands=6, lowFrequencyBound=20, highFrequencyBound=11000, neighbourRatio=0.4, staticDistribution=0.15)
    distributionshape = DistributionShape()
    energy = Energy()
    #energyband_bass, energyband_middle and energyband_high parameters come from "standard" hi-fi equalizers
    energyband_bass = EnergyBand(startCutoffFrequency=20.0, stopCutoffFrequency=150.0, sampleRate=sampleRate)
    energyband_middle_low = EnergyBand(startCutoffFrequency=150.0, stopCutoffFrequency=800.0,sampleRate=sampleRate)
    energyband_middle_high = EnergyBand(startCutoffFrequency=800.0, stopCutoffFrequency=4000.0, sampleRate=sampleRate)
    energyband_high = EnergyBand(startCutoffFrequency=4000.0, stopCutoffFrequency=12000.0,sampleRate=sampleRate)
    flatnessdb = FlatnessDB()
    flux = Flux()
    harmonic_peaks = HarmonicPeaks()
    hfc = HFC()
    mfcc = MFCC()
    rolloff = RollOff()
    rms = RMS()
    strongpeak = StrongPeak()
    erbbands= ERBBands()
    gfcc =GFCC()
    # pitch algorithms
    pitch_salience = PitchSalience()

    # dissonance
    spectral_peaks = SpectralPeaks(sampleRate=sampleRate, orderBy='frequency')
    dissonance = Dissonance()

    # spectral complexity
    # magnitudeThreshold = 0.005 is hardcoded for a "blackmanharris62" frame
    spectral_complexity = SpectralComplexity(magnitudeThreshold=0.005)



    scPool = essentia.Pool()  # pool for spectral contrast




    for frame in frames:
        #frameScope = [start_of_frame / sampleRate, (start_of_frame + frameSize) / sampleRate]
        # silence rate
        pool.add(namespace + '.' + 'silence_rate_60dB', int(essentia.isSilent(frame)))
        pool.add(namespace + '.' + 'silence_rate_30dB', int(is_silent_threshold(frame, -30)))
        pool.add(namespace + '.' + 'silence_rate_20dB', int(is_silent_threshold(frame, -20)))

        if options['skipSilence'] and essentia.isSilent(frame):
            continue
            
        # temporal descriptors
        pool.add(namespace + '.' + 'zerocrossingrate', zerocrossingrate(frame))
        (frame_lpc, frame_lpc_reflection) = lpc(frame)
        pool.add(namespace + '.' + 'temporal_lpc', frame_lpc)

        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)
        #barkbands
        frame_barkbands = barkbands(frame_spectrum)
        pool.add(namespace + '.' + 'barkbands', frame_barkbands)
        # barkbands-based descriptors
        pool.add(namespace + '.' + 'spectral_crest', crest(frame_barkbands))
        pool.add(namespace + '.' + 'spectral_flatness_db', flatnessdb(frame_barkbands))
        barkbands_centralmoments = CentralMoments(range = len(frame_barkbands) - 1)
        (barkbands_spread, barkbands_skewness, barkbands_kurtosis) = distributionshape(barkbands_centralmoments(frame_barkbands))
        pool.add(namespace + '.' + 'barkbands_spread', barkbands_spread)
        pool.add(namespace + '.' + 'barkbands_skewness', barkbands_skewness)
        pool.add(namespace + '.' + 'barkbands_kurtosis', barkbands_kurtosis)

        # spectrum-based descriptors
        power_spectrum = frame_spectrum ** 2
        pool.add(namespace + '.' + 'spectral_centroid', centroid(power_spectrum))
        pool.add(namespace + '.' + 'spectral_decrease', decrease(power_spectrum))
        pool.add(namespace + '.' + 'spectral_energy', energy(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_low', energyband_bass(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_middle_low', energyband_middle_low(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_middle_high', energyband_middle_high(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_energyband_high', energyband_high(frame_spectrum))
        pool.add(namespace + '.' + 'hfc', hfc(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_rms', rms(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_flux', flux(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_rolloff', rolloff(frame_spectrum))
        pool.add(namespace + '.' + 'spectral_strongpeak', strongpeak(frame_spectrum))

        # mfcc
        (frame_melbands, frame_mfcc) = mfcc(frame_spectrum)
        pool.add(namespace + '.' + 'mfcc', frame_mfcc)
        pool.add(namespace + '.' + 'melbands', frame_melbands)

        # erbbands
        (frame_erbbands,frame_gfcc) = gfcc(frame_spectrum)
        pool.add(namespace + '.' + 'erbbands', frame_erbbands)
        pool.add(namespace + '.' + 'gfcc', frame_erbbands)

        # spectral contrast
        (sc_coeffs, sc_valleys) = spectral_contrast(frame_spectrum)
        scPool.add(namespace + '.' + 'sccoeffs', sc_coeffs)
        scPool.add(namespace + '.' + 'scvalleys', sc_valleys)

        # pitch descriptors
        frame_pitch_salience = pitch_salience(frame_spectrum[:-1])
        pool.add(namespace + '.' + 'pitch_salience', frame_pitch_salience)

        # spectral complexity
        pool.add(namespace + '.' + 'spectral_complexity', spectral_complexity(frame_spectrum))



    #statistics

    #silence
    pool2.add(namespace + '.' + 'silence_rate_30dB_mean', np.mean(pool['lowlevel.silence_rate_30dB']))
    pool2.add(namespace + '.' + 'silence_rate_30dB_var', np.var(pool['lowlevel.silence_rate_30dB']))
    pool2.add(namespace + '.' + 'silence_rate_20dB_mean', np.mean(pool['lowlevel.silence_rate_20dB']))
    pool2.add(namespace + '.' + 'silence_rate_20dB_var', np.var(pool['lowlevel.silence_rate_20dB']))
    pool2.add(namespace + '.' + 'silence_rate_60dB_mean', np.mean(pool['lowlevel.silence_rate_60dB']))
    pool2.add(namespace + '.' + 'silence_rate_60dB_var', np.var(pool['lowlevel.silence_rate_60dB']))
    pool2.add(namespace + '.' + 'zerocrossingrate_mean', np.mean(pool['lowlevel.zerocrossingrate']))
    pool2.add(namespace + '.' + 'zerocrossingrate_var', np.var(pool['lowlevel.zerocrossingrate']))
    statistics = compute_statistics(pool['lowlevel.temporal_lpc'])
    pool2.add(namespace + '.' + 'temporal_lpc_mean', statistics[0])
    pool2.add(namespace + '.' + 'temporal_lpc_var', statistics[1])

    #barkbands
    statistics = compute_statistics(pool['lowlevel.barkbands'])
    pool2.add(namespace + '.' + 'barkbands_mean', statistics[0])
    pool2.add(namespace + '.' + 'barkbands_var', statistics[1])
    pool2.add(namespace + '.' + 'spectral_crest_mean', np.mean(pool['lowlevel.spectral_crest']))
    pool2.add(namespace + '.' + 'spectral_crest_var', np.var(pool['lowlevel.spectral_crest']))
    pool2.add(namespace + '.' + 'spectral_flatness_db_mean', np.mean(pool['lowlevel.spectral_flatness_db']))
    pool2.add(namespace + '.' + 'spectral_flatness_db_var', np.var(pool['lowlevel.spectral_flatness_db']))
    pool2.add(namespace + '.' + 'barkbands_spread_mean', np.mean(pool['lowlevel.barkbands_spread']))
    pool2.add(namespace + '.' + 'barkbands_spread_var', np.var(pool['lowlevel.barkbands_spread']))
    pool2.add(namespace + '.' + 'barkbands_kurtosis_mean', np.mean(pool['lowlevel.barkbands_kurtosis']))
    pool2.add(namespace + '.' + 'barkbands_kurtosis_var', np.var(pool['lowlevel.barkbands_kurtosis']))
    pool2.add(namespace + '.' + 'barkbands_skewness_mean', np.mean(pool['lowlevel.barkbands_skewness']))
    pool2.add(namespace + '.' + 'barkbands_skewness_var', np.var(pool['lowlevel.barkbands_skewness']))

    # spectrum-based descriptors
    pool2.add(namespace + '.' + 'spectral_centroid_mean', np.mean(pool['lowlevel.spectral_centroid']))
    pool2.add(namespace + '.' + 'spectral_centroid_var', np.var(pool['lowlevel.spectral_centroid']))
    pool2.add(namespace + '.' + 'spectral_decrease_mean', np.mean(pool['lowlevel.spectral_decrease']))
    pool2.add(namespace + '.' + 'spectral_decrease_var', np.var(pool['lowlevel.spectral_decrease']))
    pool2.add(namespace + '.' + 'spectral_energy_mean', np.mean(pool['lowlevel.spectral_energy']))
    pool2.add(namespace + '.' + 'spectral_energy_var', np.var(pool['lowlevel.spectral_energy']))
    pool2.add(namespace + '.' + 'spectral_energyband_low_mean',np.mean(pool['lowlevel.spectral_energyband_low']))
    pool2.add(namespace + '.' + 'spectral_energyband_low_var',np.var(pool['lowlevel.spectral_energyband_low']))
    pool2.add(namespace + '.' + 'spectral_energyband_middle_low_mean',np.mean(pool['lowlevel.spectral_energyband_middle_low']))
    pool2.add(namespace + '.' + 'spectral_energyband_middle_low_var',np.var(pool['lowlevel.spectral_energyband_middle_low']))
    pool2.add(namespace + '.' + 'spectral_energyband_middle_high_mean',np.mean(pool['lowlevel.spectral_energyband_middle_high']))
    pool2.add(namespace + '.' + 'spectral_energyband_middle_high_var',np.var(pool['lowlevel.spectral_energyband_middle_high']))
    pool2.add(namespace + '.' + 'spectral_energyband_high_mean', np.mean(pool['lowlevel.spectral_energyband_high']))
    pool2.add(namespace + '.' + 'spectral_energyband_high_var', np.var(pool['lowlevel.spectral_energyband_high']))
    pool2.add(namespace + '.' + 'hfc_mean', np.mean(pool['lowlevel.hfc']))
    pool2.add(namespace + '.' + 'hfc_var', np.var(pool['lowlevel.hfc']))
    pool2.add(namespace + '.' + 'spectral_rms_mean', np.mean(pool['lowlevel.spectral_rms']))
    pool2.add(namespace + '.' + 'spectral_rms_var', np.var(pool['lowlevel.spectral_rms']))
    pool2.add(namespace + '.' + 'spectral_rolloff_mean', np.mean(pool['lowlevel.spectral_rolloff']))
    pool2.add(namespace + '.' + 'spectral_rolloff_var', np.var(pool['lowlevel.spectral_rolloff']))
    pool2.add(namespace + '.' + 'spectral_strongpeak_mean', np.mean(pool['lowlevel.spectral_strongpeak']))
    pool2.add(namespace + '.' + 'spectral_strongpeakh_var', np.var(pool['lowlevel.spectral_strongpeak']))

    #mfcc
    statistics = compute_statistics(pool['lowlevel.mfcc'])
    pool2.add(namespace + '.' + 'mfcc_mean', statistics[0])
    pool2.add(namespace + '.' + 'mfcc_var', statistics[1])
    statistics = compute_statistics(pool['lowlevel.melbands'])
    pool2.add(namespace + '.' + 'melbands_mean', statistics[0])
    pool2.add(namespace + '.' + 'melbands_var', statistics[1])

    #erbbands

    statistics = compute_statistics(pool['lowlevel.erbbands'])
    pool2.add(namespace + '.' + 'erbbands_mean', statistics[0])
    pool2.add(namespace + '.' + 'erbbands_var', statistics[1])
    statistics = compute_statistics(pool['lowlevel.gfcc'])
    pool2.add(namespace + '.' + 'gfcc_mean', statistics[0])
    pool2.add(namespace + '.' + 'gfcc_var', statistics[1])


    #spectralcontrast
    spectral_contrast=np.array(spectralContrastPCA(scPool))
    pool.add(namespace + '.' + 'spectral_contrast', spectral_contrast)
    statistics = compute_statistics(spectral_contrast)
    pool2.add(namespace + '.' + 'spectral_contrast_mean', statistics[0])
    pool2.add(namespace + '.' + 'spectral_contrast_var', statistics[1])


    # pitch descriptors
    pool2.add(namespace + '.' + 'pitch_salience_mean', np.mean(pool['lowlevel.pitch_salience']))
    pool2.add(namespace + '.' + 'pitch_salience_var', np.var(pool['lowlevel.pitch_salience']))
    pool2.add(namespace + '.' + 'spectral_complexity_mean', np.mean(pool['lowlevel.spectral_complexity']))
    pool2.add(namespace + '.' + 'sspectral_complexity_var', np.var(pool['lowlevel.spectral_complexity']))

    return pool2
def compute_tonal(audio, pool2, options):
    namespace = 'tonal'
    pool = essentia.Pool()
    # analysis parameters
    sampleRate = options['sampleRate']
    frameSize  = options['frameSize']
    hopSize    = options['hopSize']

    frames = FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    window = Windowing(size = frameSize, zeroPadding = 0)
    spectrum = Spectrum(size = frameSize)
    spectral_peaks = SpectralPeaks(maxPeaks = 10000, magnitudeThreshold = 0.00001, minFrequency = 40, maxFrequency = 5000, orderBy = "frequency")
    tuning = TuningFrequency()

    # computing the tuning frequency
    tuning_frequency = 440.0

    for frame in frames:

        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)

        (frame_frequencies, frame_magnitudes) = spectral_peaks(frame_spectrum)

        #if len(frame_frequencies) > 0:
        (tuning_frequency, tuning_cents) = tuning(frame_frequencies, frame_magnitudes)

    pool2.add(namespace + '.' + 'tuning_frequency', tuning_frequency)#, pool.GlobalScope)

    # computing the HPCPs
    spectral_whitening = SpectralWhitening()

    hpcp_key_size = 36
    hpcp_chord_size = 36
    hpcp_tuning_size = 120

    hpcp_key = HPCP(size = hpcp_key_size,
                             referenceFrequency = tuning_frequency,
                             bandPreset = False,
                             minFrequency = 40.0,
                             maxFrequency = 5000.0,
                             weightType = 'squaredCosine',
                             nonLinear = False,
                             windowSize = 4.0/3.0,
                             sampleRate = sampleRate)

    hpcp_chord = HPCP(size = hpcp_chord_size,
                               referenceFrequency = tuning_frequency,
                               harmonics = 8,
                               bandPreset = True,
                               minFrequency = 40.0,
                               maxFrequency = 5000.0,
                               weightType = 'cosine',
                               nonLinear = True,
                               windowSize = 0.5,
                               sampleRate = sampleRate)

    hpcp_tuning = HPCP(size = hpcp_tuning_size,
                                referenceFrequency = tuning_frequency,
                                harmonics = 8,
                                bandPreset = True,
                                minFrequency = 40.0,
                                maxFrequency = 5000.0,
                                weightType = 'cosine',
                                nonLinear = True,
                                windowSize = 0.5,
                                sampleRate = sampleRate)

    # intializing the HPCP arrays
    hpcps_key = []
    hpcps_chord = []
    hpcps_tuning = []

    # computing HPCP loop
    frames = FrameGenerator(audio = audio, frameSize = frameSize, hopSize = hopSize)
    for frame in frames:

        if options['skipSilence'] and essentia.isSilent(frame):
          continue

        frame_windowed = window(frame)
        frame_spectrum = spectrum(frame_windowed)

        # spectral peaks
        (frame_frequencies, frame_magnitudes) = spectral_peaks(frame_spectrum)

        if (len(frame_frequencies) > 0):
           # spectral_whitening
           frame_magnitudes_white = spectral_whitening(frame_spectrum, frame_frequencies, frame_magnitudes)
           frame_hpcp_key = hpcp_key(frame_frequencies, frame_magnitudes_white)
           frame_hpcp_chord = hpcp_chord(frame_frequencies, frame_magnitudes_white)
           frame_hpcp_tuning = hpcp_tuning(frame_frequencies, frame_magnitudes_white)
        else:
           frame_hpcp_key = essentia.array([0] * hpcp_key_size)
           frame_hpcp_chord = essentia.array([0] * hpcp_chord_size)
           frame_hpcp_tuning = essentia.array([0] * hpcp_tuning_size)

        # key HPCP
        hpcps_key.append(frame_hpcp_key)

        # add HPCP to the pool
        pool.add(namespace + '.' +'hpcp', frame_hpcp_key)

        # chords HPCP
        hpcps_chord.append(frame_hpcp_chord)

        # tuning system HPCP
        hpcps_tuning.append(frame_hpcp_tuning)

    # check if silent file
    if len(hpcps_key) == 0:
       raise EssentiaError('This is a silent file!')

    # key detection
    key_detector = Key(profileType = 'temperley')
    average_hpcps_key = np.average(essentia.array(hpcps_key), axis=0)
    average_hpcps_key = normalize(average_hpcps_key)

    # thpcps
    max_arg = np.argmax( average_hpcps_key )
    thpcp=[]
    for i in range( max_arg, len(average_hpcps_key) ):
        thpcp.append( float(average_hpcps_key[i]) )
    for i in range( max_arg ):
        thpcp.append( float(average_hpcps_key[i]) )
    pool2.add(namespace + '.' +'thpcp', thpcp)#, pool.GlobalScope  )

    # tuning system features
    keydetector	= Key(profileType = 'diatonic')
    average_hpcps_tuning = np.average(essentia.array(hpcps_tuning), axis=0)
    average_hpcps_tuning = normalize(average_hpcps_tuning)
    (key, scale, diatonic_strength, first_to_second_relative_strength) = keydetector(essentia.array(average_hpcps_tuning))

    pool2.add(namespace + '.' +'tuning_diatonic_strength', diatonic_strength)#, pool.GlobalScope)

    (equal_tempered_deviation,
     nontempered_energy_ratio,
     nontempered_peaks_energy_ratio) = HighResolutionFeatures()(average_hpcps_tuning)

    pool2.add(namespace + '.' +'tuning_equal_tempered_deviation', equal_tempered_deviation)#, pool.GlobalScope)
    pool2.add(namespace + '.' +'tuning_nontempered_energy_ratio', nontempered_energy_ratio)#, pool.GlobalScope)
    pool2.add(namespace + '.' +'tuning_nontempered_peaks_energy_ratio', nontempered_peaks_energy_ratio)#, pool.GlobalScope)


    #hpcp
    statistics = compute_statistics(pool['tonal.hpcp'])
    pool2.add(namespace + '.' + 'hpcp_mean', statistics[0])
    pool2.add(namespace + '.' + 'hpcp_var', statistics[1])
    return pool2

# Method to obtain array with audio features
def compute_audio_features(audio,options):
    loader = MonoLoader(filename=audio,sampleRate=24000)
    audio = loader()
    features = compute_lowlevel(audio, options)
    features = compute_tonal(audio, features, options)
    audio_features = []
    column_labels = []
    for feature in features.descriptorNames():
        x = features[feature]
        if x.shape == (1,):
            y = [x[0]]
        else:
            y = x[0].tolist()
        c = 0
        for i in range(len(y)):
            c += 1
            z = feature + str(c)
            column_labels.append(z)
        audio_features = audio_features + y
    audio_features = np.array(audio_features)
    return audio_features, column_labels





