import numpy as np
import os
import scipy.io.wavfile
import librosa
import resampy

import configs
import utils

class SpeechManipulator(object):
    def __init__(self, source_path, output_path):
        self.source_path = source_path
        self.output_path = output_path

    def get_paths(self, output_dir_name):
        """ Returns the paths to source wav files and output wav files. """

        output_path = f'{self.output_path}{output_dir_name}'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        wav_paths = []
        output_paths = []
        files = os.listdir(self.source_path)
        for wav_file in files:
            fname = wav_file.split('.wav')[0]
            wav_paths.append(f'{self.source_path}{wav_file}')
            output_paths.append(f'{output_path}/{fname}.wav')
        return wav_paths, output_paths

    def sinewave_speech(self):
        '''
        Adapted from https://www.ee.columbia.edu/~dpwe/resources/matlab/sws/
        and
        https://github.com/kastnerkyle/tools/blob/master/audio/audio_tools.py

        Check out:
        http://signalsprocessed.blogspot.com/2016/08/audio-resampling-in-python.html
        '''

        wav_paths, output_paths = self.get_paths('sinewave')
        for wav_path, output_path in zip(wav_paths, output_paths):
            sr, X = scipy.io.wavfile.read(wav_path)
            #X, sr = librosa.load(wav_path, sr=8000)
            X = X.astype('float32') / (2 ** 15) # From kk?
            if len(X.shape) == 2: # In case of stereo sound
                X = X[:, 0]

            X = resampy.resample(X, sr, 8000)
            sr = 8000

            freq, magnit = utils.sinusoid_analysis(X, input_sample_rate=sr)
            X_sine = utils.sinusoid_synthesis(freq, magnit)
            scipy.io.wavfile.write(output_path, sr, utils.soundsc(X_sine))

if __name__ == "__main__":
    sm = SpeechManipulator(configs.speech_dir, configs.manipulated_speech_dir)
    sm.sinewave_speech()
