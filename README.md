# Auditory CNN and SCZ
What (schizophrenia-like) mechanisms drive auditory hallucinations?

# Usage
Path to data files should be provided in `src/data/configs.py`

# Dependencies
Besides default anaconda packages...
Python packages: tensorflow (1.15.0), librosa, [pycochleagram](https://github.com/mcdermottLab/pycochleagram), [tfcochleagram](https://github.com/jenellefeather/tfcochleagram)

# Installation Notes
- The `pycochleagram` demo requires pyaudio, which is a pain to install on Mac M1 due to some weirdness with portaudio. I use python-sounddevice as an alternative, and replace the utils.play\_array function.
- Add a setup.py file into tfcochleagram and install it as a package in your virtual environment.
- There is an incompatability with loading .npy files saved in Python 2 when using Python 3. I've written a script `convert_to_pickle3.py` that generates Python3-protocol pickle files from Python2 .npy files. If we later want compatability between Python 2 and 3, we can easily specify `protocol=2` during `pickle.dump`. Note that I correspondingly alter the code in `kelletal2018` to load the new pickle files.

# Tensorflow Installation Notes
- Install version 1.15.0
- If you have Python 3.9 you will need to set up your virtual environment to 3.7 or lower, or tensorflow will not work.
- Install tensorflow with conda or jupyter notebook will be angry.
- You may need to downgrade tensorflow estimator to match the tensorflow version: `pip install tensorflow-estimator==1.15.1`
