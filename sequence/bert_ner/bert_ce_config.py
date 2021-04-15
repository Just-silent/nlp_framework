# -*- coding: utf-8 -*-
# @Author   : Just-silent
# @time     : 2020/10/26 9:53

import dynamic_yaml
import pkg_resources

from common.config.common_config import CommonConfig

class BertConfig(CommonConfig):

    def __init__(self, config_file):
        super(BertConfig, self).__init__()
        self._config_file = config_file
        pass

    def load_config(self):
        with pkg_resources.resource_stream("sequence.bert_ner", self._config_file) as res:
            config = dynamic_yaml.load(res)
        self._config.update(config)
        return self._config
        pass


if __name__ == '__main__':
    config_file = 'bert_ce_config.yml'
    ee_config = BertConfig(config_file)
    config = ee_config.load_config()
