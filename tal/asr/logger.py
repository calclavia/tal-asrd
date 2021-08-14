from pytorch_lightning.logging import WandbLogger, rank_zero_only
import wandb
from .data import DEFAULT_SR

class WandbLoggerWrapper(WandbLogger):
    # Fixes a missing feature from lightning

    def __init__(self, **kwargs):
        self._entity = kwargs.get('entity')
        self._args = kwargs.get('args')
        self.watching_model = False
        if self._entity:
            del kwargs['entity']
        del kwargs['args']
        super().__init__(**kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_experiment'] = None
        return state

    @rank_zero_only
    def init(self, model):
        if not self.watching_model:
            # A hack to initialize the experiment
            print(self.experiment)
            wandb.watch(model, log='parameters')
            self.watching_model = True

    @rank_zero_only
    def update_config(self, data):
        """ Updates config with data """
        wandb.config.update(data)

    @property
    def experiment(self):
        r"""
          Actual wandb object. To use wandb features do the following.
          Example::
              self.logger.experiment.some_wandb_function()
          """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"
            self._experiment = wandb.init(
                name=self._name, dir=self._save_dir, project=self._project, anonymous=self._anonymous,
                id=self._id, resume="allow", tags=self._tags, entity=self._entity, config=self._args)
        return self._experiment

    @rank_zero_only
    def log_generation(self, audio, ref_text, hyp_text):
        wandb.log({"audio": [wandb.Audio(audio, caption=ref_text, sample_rate=DEFAULT_SR)]})

        if ref_text is not None and hyp_text is not None:
            data = [ref_text, hyp_text]
            wandb.log({"asr": wandb.Table(data=data, columns=["Text"])})

    @rank_zero_only
    def log_lm_generation(self, ref_text, hyp_text):
        if ref_text is not None and hyp_text is not None:
            data = [ref_text, hyp_text]
            wandb.log({"asr": wandb.Table(data=data, columns=["Text"])})

    @property
    def name(self):
        return 'WandbLogger'

    @property
    def version(self):
        return '1.0'
