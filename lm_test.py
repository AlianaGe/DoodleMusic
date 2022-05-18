from doodle_music import *

bundle = sequence_generator_bundle.read_bundle_file(MUSIC_RNN)
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map[MUSIC_RNN_TYPE](checkpoint=None, bundle=bundle)
melody_rnn.initialize()