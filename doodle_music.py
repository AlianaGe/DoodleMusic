import os
import numpy as np
from doodle_musegan import musegan
import note_seq
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2

PROJECT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(PROJECT_DIR, 'resources')
DATA_MIDI = os.path.join(DATA_DIR, 'midi', 'in')
DATA_PIANOROLL = os.path.join(DATA_DIR, 'pianorolls')
SOUND_FONT_PATH = os.path.join(DATA_DIR, 'soundfont')

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
                                               'fake_x_hard_thresholding')
RESULT_WAV = os.path.join(RESULT_DIR, 'wav')
RESULT_ACCOMPANY_NPY = os.path.join(RESULT_DIR, 'pianorolls', 'fake_x_hard_thresholding')

bundle = sequence_generator_bundle.read_bundle_file(MUSIC_RNN)
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map[MUSIC_RNN_TYPE](checkpoint=None, bundle=bundle)
melody_rnn.initialize()


def duet(savepath, midipath=os.path.join(DATA_MIDI, 'twinkle12.mid'), next_steps="default", temperature=1, ):
    # Initialize the model.
    input_sequence = note_seq.midi_file_to_note_sequence(midipath)
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


def duet_whole(savepath, midipath=os.path.join(DATA_MIDI, 'twinkle12.mid'), next_steps="default", temperature=1, ):
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

    input_sequence = note_seq.midi_file_to_note_sequence(midipath)

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


def midi_to_condition_track(midipath):
    """ midipath(single track) to pypianoroll (1,-1...)"""
    import pypianoroll
    # condition track:[4,48,84,1]
    n_bar = 4
    beat_resolution = 12
    n_timesteps = n_bar * beat_resolution * 4  # 4/4 拍
    n_pitch = 84

    pr = pypianoroll.Multitrack(filepath=midipath,
                                beat_resolution=beat_resolution)
    # pr.clip(24, 108)  # C1 -> C8
    c = np.array(pr.tracks[0].pianoroll, dtype=int)[:, 24:108]
    # change to -1,1 as input of musegan
    c[c == c.min()] = -1
    c[c == c.max()] = 1
    c = c.reshape(n_bar, 4 * beat_resolution, n_pitch, -1)
    return c


def accompany_result_to_midi(savepath, npy):
    import pypianoroll
    velocity = [60, 100, 20, 20, 30]
    programs = [127, 0, 46, 33, 48]  # [0, 0, 25, 33, 48] # drum,piano,harp/guitar,bass,string
    is_drums = [True, False, False, False, False]  # [1, 0, 0, 0, 0]
    tracks = []
    a = npy
    # timestep*pitch
    for i in range(5):
        b = a[..., i]
        b = b.reshape(-1, 84)
        b[b > 0] = velocity[i]
        b = np.pad(b, ((0, 0), (24, 20)), constant_values=-1)
        track = pypianoroll.Track(b, program=programs[i], is_drum=is_drums[i])
        tracks.append(track)

    pr = pypianoroll.Multitrack(tracks=tracks, tempo=120, beat_resolution=12)
    pr.write(savepath)


def accompany(savepath, midipath):
    import pypianoroll
    # condition track:[4,48,84,1]
    n_bar = 4
    beat_resolution = 12
    n_timesteps = n_bar * beat_resolution * 4  # 4/4 拍
    n_pitch = 84

    pr = pypianoroll.Multitrack(filepath=midipath,
                                beat_resolution=beat_resolution)
    # pr.clip(24, 108)  # C1 -> C8
    c = np.array(pr.tracks[0].pianoroll, dtype=int)[:, 24:108]
    # change to -1,1 as input of musegan
    c[c == c.min()] = -1
    c[c == c.max()] = 1
    # 204*84
    n_submidi = int(np.ceil(c.shape[0] / n_timesteps))  # 向上取整
    # 不足补0
    tmp = -np.ones([n_submidi * n_timesteps - c.shape[0], n_pitch], dtype=int)
    c = np.append(c, tmp, axis=0)
    result = []
    for i in range(n_submidi):
        cc = c[i * n_timesteps:(i + 1) * n_timesteps, :].reshape(n_bar, 4 * beat_resolution, n_pitch, -1)
        result.append(musegan(cc, RESULT_DIR, str(i))[0])  # musegan返回[1,4,48,84,1]的
    result = np.array(result)
    accompany_result_to_midi(savepath, result)
    print("Accompany Done!")


def main():
    # accompany(os.path.join(DATA_MIDI, 'galaxy8.mid'), os.path.join(RESULT_MIDI_ACCOMPANY, 'galaxyAcc.mid'))
    # duet(os.path.join(RESULT_MIDI_DUET, 'lala-2.mid'), midipath=os.path.join(DATA_MIDI, 'lala.mid'))
    # wav_to_sf2(os.path.join(DATA_DIR,'wav','test.wav'),os.path.join(SOUND_FONT_PATH,'3090.sf2'))
    # accompany(os.path.join(RESULT_MIDI_DUET, 'lala-1.mid'), os.path.join(RESULT_MIDI_ACCOMPANY, 'lalaAcc.mid'))
    # accompany(os.path.join(DATA_MIDI, 'galaxy16.mid'), os.path.join(RESULT_MIDI_ACCOMPANY, 'galaxy16Acc.mid'))
    # duet_whole('1d.mid', midipath='./1.mid')

    # duet_whole('./test_midi/lala8lkbk.mid', midipath='./test_midi/lala4-1.mid')
    # duet('./test_midi/lala4-4.mid', midipath='./test_midi/lala4-3.mid')
    # duet_whole('./test_midi/galaxy4att.mid', midipath='./test_midi/galaxy4.mid')

    # accompany('./test_midi/lala8-2.mid', './test_midi/lala8-2Acc.mid')
    accompany('./test_midi/testAcc.mid', './test_midi/test.mid')
    # accompany('./test_midi/1.mid', './test_midi/lala8-1.mid')
    # accompany('./test_midi/1.mid', './test_midi/galaxy8.mid', )
    # accompany_result_to_midi(os.path.join(RESULT_DIR, 'pianorolls', 'fake_x_hard_thresholding',
    #                                       'fake_x_hard_thresholding_h.npz'), './test_midi/1.mid')
    # t1()


if __name__ == "__main__":
    main()
