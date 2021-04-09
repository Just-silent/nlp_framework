#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File     : common_config
# @Author   : Xiaoming Liu
# @Time     : 2020/10/01 10:00
import dynamic_yaml
import pkg_resources
from abc import ABC, abstractmethod


class CommonConfig(ABC):

    def __init__(self):
        super(CommonConfig, self).__init__()
        self._config_file = "common_config.yml"
        self._config = None
        self.load_common_config()
        pass

    def load_common_config(self):
        with pkg_resources.resource_stream("common.config", self._config_file) as res:
            self._config = dynamic_yaml.load(res)
        return self._config

    @abstractmethod
    def load_config(self):
        pass

    pass
