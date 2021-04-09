'''
Author: WEY
Date: 2020-12-28 09:23:43
LastEditTime: 2020-12-28 11:18:23
'''
import torch.nn as nn

class DecodeLayer(nn.Module):
    pass
class DAE(nn.Module):
    def __init__(self, dim_embedding, dim_hidden, num_layers, num_vocab, dropout_rate, dropout):
        super(DAE, self).__init__()

        self._dim_embedding = dim_embedding
        self._dim_hidden = dim_hidden
        self._num_layers = num_layers
        self._num_vocab = num_vocab

        self._dropout_rate = dropout_rate
        # self._dropout = nn.Dropout(self._dropout_rate)
        self._lstm_dropout = dropout

        self._lstm = nn.LSTM(
            input_size=self._dim_embedding, hidden_size=self._dim_hidden // 2, batch_first=True,
            bidirectional=True, num_layers=self._num_layers, dropout=self._dropout_rate
        )
        self._lm_decoder = nn.Linear(self._dim_hidden, self._num_vocab)

    def forward(self, embed, sent_lengths):
        lstm_output = self.encode(embed=embed, sent_lengths=sent_lengths)
        lm_output = self.decode(src_encoding=lstm_output)

        return lm_output
        pass

    def encode(self, embed, sent_lengths):
        packed_src_embed = pack_padded_sequence(embed, sent_lengths, batch_first=True)
        _, hidden = self._lstm(packed_src_embed)
        lstm_output, _ = self._lstm(packed_src_embed, hidden)
        lstm_output = pad_packed_sequence(lstm_output)
        lstm_output = self._lstm_dropout(lstm_output[0])

        return lstm_output
        pass

    def decode(self, src_encoding):
        decoded = self._lm_decoder(
            src_encoding.contiguous().view(src_encoding.size(0) * src_encoding.size(1), src_encoding.size(2)))
        lm_output = decoded.view(src_encoding.size(0), src_encoding.size(1), decoded.size(1))

        return lm_output
        pass
