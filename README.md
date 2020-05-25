
# DCASE 2020 - TASK 3 - APRI

This is our contribution to [DCASE2020 Task 3 Challenge]((http://dcase.community/challenge2020/task-sound-event-localization-and-detection)).
(C) 2020 Andrés Pérez-López and Rafael Ibañez-Usach.

## Repo Structure

 * **APRI** contains the files created in our contribution
 * **seld-dcase2020** contains a fork of the [baseline system](https://github.com/sharathadavanne/seld-dcase2020), with minimal adaptations for our usage.

## Dependencies
Probably a lot.
* Ipython (why?)
* soundfile (pysoundfile)
* librosa
* keras (I'm using v2.3.1)
* tensorflow (I'm currently using v2.0.0, since newest 2.1+ collides with that keras version)
* ...

## Getting started
1. Download the dataset, and place it in a suitable path in your computer
2. Go to `baseline/parameter.py` and change the paths at the required places (marked with a TODO sign) 
3. Run `generate_audio_from_annotations.py`. This will give as output a lot of monophonic signals to start playing with.

To run the baseline system:
1. Run `baseline/batch_feature_extraction.py` in `dev` mode, so you can preprocess the data and create the spectrograms and stuff on the `/feat_label` folder.
2. (optional) Run `baseline/seld.py` passing a 4 as parameter (this is the preset number, configurable in `parameter.py`). This will make a quick training of the baseline method on the foa dataset. 

## License
Inside **seld-dcase2020** folder: 
Except for the contents in the `metrics` folder that have [MIT License](seld-dcase2020/metrics/LICENSE.md). The rest of the repository is licensed under the [TAU License](seld-dcase2020/LICENSE.md).
See 

Otherwise: [Do What the Fuck You Want To Public License](APRI/LICENSE.md)


