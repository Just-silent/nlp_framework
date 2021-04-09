from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """
    Abstract definition for runner
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _build_config(self):
        pass

    @abstractmethod
    def _build_data(self):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_loss(self):
        pass

    @abstractmethod
    def _build_optimizer(self):
        pass

    @abstractmethod
    def _build_evaluator(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def _train_epoch(self, episode):
        pass

    @abstractmethod
    def _valid(self, episode, valid_log_writer):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def _display_result(self, episode):
        pass

    @abstractmethod
    def _save_checkpoint(self, epoch):
        pass

    @abstractmethod
    def _load_checkpoint(self):
        pass
