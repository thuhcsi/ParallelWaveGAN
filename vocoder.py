
import os

import torch
import yaml
from scipy.io import wavfile

from parallel_wavegan.utils import load_model


class Vocoder:
    def __init__(self, checkpoint, config=None):
        """
        Parameters
        ----------
            checkpoint: str, the path of model checkpoint file.
            config: str, the path of model configuration file.
        """

        # load config
        if config is None:
            dirname = os.path.dirname(checkpoint)
            config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            self._config = yaml.load(f, Loader=yaml.Loader)

        # setup model
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._model = load_model(checkpoint, self._config)
        self._model.remove_weight_norm()
        self._model = self._model.eval().to(self._device)


    def mel2wav(self, mel, output_file=None):
        """
        Parameters
        ----------
            mel: numpy.ndarray of mel spectrogram [shape=(#spec_frame, @n_mels)]
            output_file: file path name to write the generated wav data

        Returns
        -------
            wav: numpy.ndarray of generated waveform in [-1, 1] [shape=(#sample_point,)]
        """
        with torch.no_grad():
            mel = torch.tensor(mel, dtype=torch.float).to(self._device)
            wav = self._model.inference(mel).view(-1).cpu().numpy()

        if output_file:
            wavfile.write(output_file, self._config['sampling_rate'], (wav * 32767).astype('int16'))
        return wav
