'''
Author: WEY
Date: 2020-12-29 17:13:06
LastEditTime: 2021-01-06 09:48:16
'''
from common.model.embeder_layer import EmbedLayer
from common.model.encoder_layer import EncodeLayer
from common.model.decoder_layer import DecodeLayer
from common.model.common_model import CommonModel
class GenericModel(CommonModel):
    """
    docstring
    """

    def __init(self, config):
        super(GenericModel,self).__init__(config)
        self._embeder = EmbedLayer(config) 
        self._encoder = EncodeLayer(config)
        self._decoder  = DecodeLayer(config)

    def forward(self,*inputs):
        embed = self._embeder(inputs) # 256,15,350
        encoded = self._encoder(embed) 
        outputs =self._decoder(encoded) 
        return outputs