import os
import numpy as np

PROJECT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(PROJECT_DIR, 'resources')
DATA_MIDI = os.path.join(DATA_DIR, 'midi', 'in')
DATA_PIANOROLL = os.path.join(DATA_DIR, 'pianorolls')
SOUND_FONT_PATH = os.path.join(DATA_DIR, 'soundfont')

# MUSIC_RNN = os.path.join(DATA_DIR, 'model_checkpoints', 'music_rnn', 'lookback_rnn.mag')
MUSIC_RNN_TYPE = 'basic_rnn'
MUSIC_RNN = os.path.join(DATA_DIR, 'model_checkpoints', 'music_rnn', MUSIC_RNN_TYPE + '.mag')
# MODEL_DIR_MUSEGAN = os.path.join(DATA_DIR, 'model_checkpoints', 'musegan')

RESULT_DIR = DATA_DIR
RESULT_MIDI = os.path.join(RESULT_DIR, 'midi', 'out')
RESULT_MIDI_ACCOMPANY = os.path.join(RESULT_MIDI, 'accompany')
RESULT_MIDI_DUET = os.path.join(RESULT_MIDI, 'duet')
RESULT_ACCOMPANY_PIANOROLL_B = os.path.join(RESULT_DIR, 'pianorolls', 'fake_x_bernoulli_sampling',
                                            'fake_x_bernoulli_sampling_0.npz')
RESULT_ACCOMPANY_PIANOROLL_HARD = os.path.join(RESULT_DIR, 'pianorolls', 'fake_x_hard_thresholding',
                                               'fake_x_hard_thresholding_0.npz')
RESULT_WAV = os.path.join(RESULT_DIR, 'wav')


def duet(savepath, next_steps="default", temperature=1, midifile=os.path.join(DATA_MIDI, 'twinkle12.mid')):
    import note_seq
    from magenta.models.melody_rnn import melody_rnn_sequence_generator
    from magenta.models.shared import sequence_generator_bundle
    from note_seq.protobuf import generator_pb2

    # Initialize the model.
    print("Initializing Melody RNN...")
    bundle = sequence_generator_bundle.read_bundle_file(MUSIC_RNN)
    generator_map = melody_rnn_sequence_generator.get_generator_map()
    melody_rnn = generator_map[MUSIC_RNN_TYPE](checkpoint=None, bundle=bundle)
    melody_rnn.initialize()
    print('🎉 Done!')

    input_sequence = note_seq.midi_file_to_note_sequence(midifile)

    # Set the start time to begin on the next step after the last note ends.
    qpm = input_sequence.tempos[0].qpm
    last_end_time = (max(n.end_time for n in input_sequence.notes)
                     if input_sequence.notes else 0)
    seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter  # 原曲总时长
    ori_steps = last_end_time / seconds_per_step  # 总step
    if (next_steps == "default"):
        next_steps = ori_steps

    num_steps = ori_steps + next_steps
    total_seconds = num_steps * seconds_per_step
    generator_options = generator_pb2.GeneratorOptions()
    generator_options.args['temperature'].float_value = temperature
    generate_section = generator_options.generate_sections.add(
        start_time=last_end_time,
        end_time=total_seconds)

    # Ask the model to continue the sequence.
    sequence = melody_rnn.generate(input_sequence, generator_options)
    sequence_trimmed = note_seq.extract_subsequence(sequence, start_time=last_end_time, end_time=total_seconds)
    note_seq.note_sequence_to_midi_file(sequence_trimmed, savepath)
    return sequence_trimmed


def duet_whole(savepath, next_steps="default", temperature=1, midifile=os.path.join(DATA_MIDI, 'twinkle12.mid')):
    import note_seq
    from magenta.models.melody_rnn import melody_rnn_sequence_generator
    from magenta.models.shared import sequence_generator_bundle
    from note_seq.protobuf import generator_pb2

    # Initialize the model.
    print("Initializing Melody RNN...")
    bundle = sequence_generator_bundle.read_bundle_file(MUSIC_RNN)
    generator_map = melody_rnn_sequence_generator.get_generator_map()
    melody_rnn = generator_map[MUSIC_RNN_TYPE](checkpoint=None, bundle=bundle)
    melody_rnn.initialize()
    print('🎉 Done!')

    input_sequence = note_seq.midi_file_to_note_sequence(midifile)

    # Set the start time to begin on the next step after the last note ends.
    qpm = input_sequence.tempos[0].qpm
    last_end_time = (max(n.end_time for n in input_sequence.notes)
                     if input_sequence.notes else 0)
    seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter  # 原曲总时长
    ori_steps = last_end_time / seconds_per_step  # 总step
    if (next_steps == "default"):
        next_steps = ori_steps

    num_steps = ori_steps + next_steps
    total_seconds = num_steps * seconds_per_step
    generator_options = generator_pb2.GeneratorOptions()
    generator_options.args['temperature'].float_value = temperature
    generate_section = generator_options.generate_sections.add(
        start_time=last_end_time,
        end_time=total_seconds)

    # Ask the model to continue the sequence.
    sequence = melody_rnn.generate(input_sequence, generator_options)
    note_seq.note_sequence_to_midi_file(sequence, savepath)
    return sequence


def midi_to_condition_track(midifile):
    """ midifile(single track) to pypianoroll (1,-1...)"""
    import pypianoroll
    # condition track:[4,48,84,1]
    n_bar = 4
    beat_resolution = 12
    n_timesteps = n_bar * beat_resolution * 4  # 4/4 拍
    n_pitch = 84

    pr = pypianoroll.Multitrack(filepath=midifile,
                                beat_resolution=beat_resolution)
    # pr.clip(24, 108)  # C1 -> C8
    c = np.array(pr.tracks[0].pianoroll, dtype=int)[:, 24:108]
    # change to -1,1 as input of musegan
    c[c == c.min()] = -1
    c[c == c.max()] = 1
    c = c.reshape(n_bar, 4 * beat_resolution, n_pitch, -1)
    return c


def accompany_result_to_midi(npz, savepath):
    """ convert accompany.npz to midifile """
    import pypianoroll
    from scipy.sparse.csc import csc_matrix
    import json

    def reconstruct_sparse(target_dict, name):
        """Return a reconstructed instance of `scipy.sparse.csc_matrix`."""
        return csc_matrix((target_dict[name + '_csc_data'],
                           target_dict[name + '_csc_indices'],
                           target_dict[name + '_csc_indptr']),
                          shape=target_dict[name + '_csc_shape']).toarray()

    with np.load(npz) as loaded:
        if 'info.json' not in loaded:
            raise ValueError("Cannot find 'info.json' in the .npz file")
        info_dict = json.loads(loaded['info.json'].decode('utf-8'))
        name = info_dict['name']
        beat_resolution = info_dict['beat_resolution']

        tempo = loaded['tempo']
        if 'downbeat' in loaded.files:
            downbeat = loaded['downbeat']
        else:
            downbeat = None

        idx = 0
        tracks = []
        track_velocity = [60, 100, 0, 0, 35]  # drum, piano, guitar(×), bass, strings

        while str(idx) in info_dict:
            pianoroll = reconstruct_sparse(loaded,
                                           'pianoroll_{}'.format(idx))
            a = np.array(pianoroll, dtype=int)
            a[a > 0] = track_velocity[idx]
            track = pypianoroll.Track(a, info_dict[str(idx)]['program'],  # program是可变的！
                                      info_dict[str(idx)]['is_drum'],
                                      info_dict[str(idx)]['name'])
            tracks.append(track)
            idx += 1

    pr = pypianoroll.Multitrack(tracks=tracks, tempo=tempo, beat_resolution=beat_resolution)
    pr.write(savepath)


def to_sf2(data, filepath, sample_rate=44100, root_key=60):
    import doosf2 as doo
    doo.add_sf2(data, filepath, sample_rate, root_key)


def wav_to_sf2(wavfile, filepath, sample_rate=44100, root_key=60):
    import doosf2 as doo
    import wave
    f = wave.open(wavfile, 'r')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.frombuffer(str_data, dtype=np.int16)
    # print(wave_data)
    doo.add_sf2(wave_data, filepath, sample_rate, root_key)


def accompany(midipath, savepath):
    c = midi_to_condition_track(midipath)
    from doodle_musegan import musegan
    musegan(c, RESULT_DIR)
    accompany_result_to_midi(RESULT_ACCOMPANY_PIANOROLL_HARD, savepath)


def main():
    # accompany(os.path.join(DATA_MIDI, 'galaxy8.mid'), os.path.join(RESULT_MIDI_ACCOMPANY, 'galaxyAcc.mid'))
    # duet(os.path.join(RESULT_MIDI_DUET,'lala-2.mid'),midifile=os.path.join(DATA_MIDI,'lala.mid'))
    # wav_to_sf2(os.path.join(DATA_DIR,'wav','test.wav'),os.path.join(SOUND_FONT_PATH,'3090.sf2'))
    accompany(os.path.join(RESULT_MIDI_DUET, 'lala-1.mid'), os.path.join(RESULT_MIDI_ACCOMPANY, 'lalaAcc.mid'))
    # accompany(os.path.join(DATA_MIDI, 'galaxy16.mid'), os.path.join(RESULT_MIDI_ACCOMPANY, 'galaxy16Acc.mid'))
    # duet_whole(os.path.join(DATA_MIDI, '1-.mid'), midifile='./1.mid')


if __name__ == "__main__":
    main()
