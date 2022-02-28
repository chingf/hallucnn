import numpy as np
import os
import matplotlib.pyplot as plt
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
            X, sr = librosa.load(wav_path, sr=8000)
            analysis_sample_rate = sr

            X = X.astype('float32') / (2 ** 15) # From kk?
            if len(X.shape) == 2: # In case of stereo sound
                X = X[:, 0]

            freq, magnit, a, g, e = utils.sinusoid_analysis(
                X, input_sample_rate=sr,
                analysis_sample_rate=analysis_sample_rate
                )
            X_sine_lpc = utils.lpc_synthesis(
                a, g,
                residual_excitation=None,
                voiced_frames=None, window_step=128, emphasis=0.9
                )
            X_sine = utils.sinusoid_synthesis(
                freq, magnit, input_sample_rate=analysis_sample_rate
                )
            scipy.io.wavfile.write(output_path, sr, utils.soundsc(X_sine))

            plt.subplot(311)
            plt.specgram(X, Fs=sr)
            plt.subplot(312)
            plt.specgram(X_sine, Fs=sr)
            plt.subplot(313)
            plt.specgram(X_sine_lpc, Fs=sr)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.suptitle(output_path)
            plt.show()

if __name__ == "__main__":
    sm = SpeechManipulator(configs.speech_dir, configs.manipulated_speech_dir)
    sm.sinewave_speech()
