# Auditory CNN and SCZ
What (schizophrenia-like) mechanisms drive auditory hallucinations?

# Usage
Path to data files should be provided in `src/data/configs.py`

# Dependencies
Besides default anaconda packages...
Python packages: tensorflow, librosa, resampy,
[pycochleagram](https://github.com/mcdermottLab/pycochleagram),
[tfcochleagram](https://github.com/jenellefeather/tfcochleagram)

# Notes
- The pycochleagram demo requires pyaudio, which is a pain to install on Mac M1.
I use python-sounddevice as an alternative, and replace the utils.play\_array function.
- Add a setup.py file into tfcochleagram and install it as a package
