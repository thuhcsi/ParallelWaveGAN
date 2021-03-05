import torch
import yaml
from parallel_wavegan.utils import load_model


class Vocoder:
    def __init__(self, checkpoint, config):
        """
        
        Parameters
        ----------
        checkpoint: str, the path of model checkpoint file.
        config: str, the path of model configuration file.

        """
        self.model = load_model(checkpoint).to("cuda").eval()
        self.model.remove_weight_norm()
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)            

    def mel2wav(self, mel):
        """
        Parameters
        ----------
        mel: numpy.ndarray of mel spectrogram
        """
        with torch.no_grad():
            return self.model.inference(mel).view(-1).cpu().numpy()