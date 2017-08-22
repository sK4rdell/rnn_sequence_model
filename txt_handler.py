import numpy as np

class DataGenerator:
    def __init__(self, data_path):
        self.raw_data = self._read_data(data_path)

        chars = list(set(self.raw_data))
        self.num_chars = len(chars)
        self._decode_dir = {ix: char for ix, char in enumerate(chars)}
        self._encode_dir = {char: ix for ix, char in enumerate(chars)}
        # get encoded data, i.e. from chars to ints
        self._encoded_data = np.array(self.encode_data(self.raw_data))

    def encode_data(self, encoded_data):
        return [self._encode_dir[x] for x in encoded_data]

    def decode_txt(self, decoded_data):
        txt_list =  [self._decode_dir[x] for x in decoded_data]
        txt = ''.join(txt_list)
        return txt

    def _read_data(self, path):
        with open(path, 'r') as txt_file:
            data = txt_file.read()
        return data

    def batch_generator(self, batch_size, sequence_length, nb_epochs):

        data_len = self._encoded_data.shape[0]
        # num batches that fit in the data-set
        nb_batches = np.floor(data_len / (batch_size * sequence_length))
        usage_data_len = nb_batches * batch_size * sequence_length
        xdata = np.reshape(self._encoded_data[0:usage_data_len], [batch_size, nb_batches * sequence_length])
        ydata = np.reshape(self._encoded_data[1:usage_data_len + 1], [batch_size, nb_batches * sequence_length])

        while "It ain't over till it's over":
            for batch in range(nb_batches):
                reset_state = True if batch == 0 else False
                x = xdata[:, batch * sequence_length:(batch + 1) * sequence_length]
                y = ydata[:, batch * sequence_length:(batch + 1) * sequence_length]
                yield x, y,  reset_state
