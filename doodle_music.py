import copy
import os
import numpy as np
import random

import pretty_midi

import chorderator as cdt
import note_seq
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf_config=tf.compat.v1.ConfigProto()
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 分配 50%
tf_config.gpu_options.allow_growth = True # 自适应
session = tf.compat.v1.Session(config=tf_config)

# PROJECT_DIR = os.path.dirname(os.getcwd())
PROJECT_DIR = r'D:\MyFiles\FinalTest\music'
DATA_DIR = os.path.join(PROJECT_DIR, 'resources')
DATA_MIDI = os.path.join(DATA_DIR, 'midi', 'in')
DATA_PIANOROLL = os.path.join(DATA_DIR, 'pianorolls')
SOUND_FONT_PATH = os.path.join(DATA_DIR, 'soundfont')

MUSIC_RNN_TYPE = 'attention_rnn'
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

# cdt.load_data()


def change_tempo(note_sequence, new_tempo):
    new_sequence = copy.deepcopy(note_sequence)
    ratio = note_sequence.tempos[0].qpm / new_tempo
    for note in new_sequence.notes:
        note.start_time = note.start_time * ratio
        note.end_time = note.end_time * ratio
    new_sequence.tempos[0].qpm = new_tempo
    return new_sequence

def duet(savepath, midipath, temperature=0.9, ):
    # Initialize the model.
    input_sequence = note_seq.midi_file_to_note_sequence(midipath)
    # 原曲
    input_notes_num = len(input_sequence.notes)
    qpm = input_sequence.tempos[0].qpm
    last_end_time = (max(n.end_time for n in input_sequence.notes)
                     if input_sequence.notes else 0)
    seconds_per_step = 60.0 / qpm / melody_rnn.steps_per_quarter  # 原曲总时长
    ori_steps = last_end_time / seconds_per_step  # 总step
    # some tiny change
    if last_end_time < 4.1:
        next_steps = max(ori_steps,32)
    else:
        next_steps =  min(ori_steps,64)
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

    duet_last_end_time = (max(n.end_time for n in sequence_trimmed.notes)
                     if sequence_trimmed.notes else 0)
    duet_input_notes_num = len(sequence_trimmed.notes)
    sequence_trimmed = note_seq.extract_subsequence(sequence_trimmed, start_time=sequence_trimmed.notes[0].start_time,
                                                    end_time=total_seconds)
    if duet_last_end_time/duet_input_notes_num > 0.35:  # 快速 0.2, 慢速0.5
        sequence_trimmed = change_tempo(sequence_trimmed, max(qpm, 360))
    if duet_last_end_time/duet_input_notes_num > 0.2:  # 1/
        sequence_trimmed = change_tempo(sequence_trimmed, max(qpm, 240))

    # start = (min(n.start_time for n in sequence_trimmed.notes)
    #                  if sequence_trimmed.notes else 0)
    note_seq.note_sequence_to_midi_file(sequence_trimmed, savepath)

    return sequence_trimmed

def beat_tap_wav_write_pretty_midi(savepath, midipath):
    import pretty_midi
    import aubio
    import os
    pm = pretty_midi.PrettyMIDI(midipath)
    os.system('fluidsynth -F 1.wav ' + midipath)
    # -----------------
    win_s = 1024  # fft size
    hop_s = win_s // 2  # hop size: number of samples read per iteration
    a_source = aubio.source('./1.wav')
    samplerate = a_source.samplerate
    a_tempo = aubio.tempo("default", win_s, hop_s, samplerate)
    tap = pretty_midi.Instrument(program=117)
    # 1s:44100 points
    total_read = 0
    flag = 1
    while True:
        samples, read = a_source()  # read 点数
        total_read += read
        is_beat = a_tempo(samples)
        if is_beat and flag > 0:
            t = total_read / samplerate
            new_note = pretty_midi.Note(127, 60, t, t + 0.5)
            tap.notes.append(new_note)
        if read < a_source.hop_size:
            break
        flag = -flag

    pm.instruments.append(tap)
    # Write out the MIDI data
    pm.write(savepath)
    print("Beat Tracking Done")


def accompany(savepath, midipath):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_file=midipath)
    start = pm.instruments[0].notes[0].start
    end = pm.get_end_time()
    new_end = end - start
    pm.adjust_times([start, end], [0, new_end])
    n_submidi = int(np.ceil(new_end / 4))
    pm.instruments[0].notes[-1].end = n_submidi * 8
    pm.write('tmp1.mid')
    # pm = pretty_midi.PrettyMIDI('tmp1.mid')
    cdt.set_melody('tmp1.mid')
    cdt.set_meta(tonic=cdt.Key.C, meter='4/4')
    cdt.set_segmentation('A4' * n_submidi)
    cdt.set_output_style(cdt.Style.POP_STANDARD)
    gen = cdt.generate_save(savepath)

    velocities = [100, 30, 5]
    idx = 0
    for i in gen.instruments:
        tmpnotes = []
        for n in i.notes:
            n.velocity = velocities[idx]
            if n.end > new_end:
                n.end = new_end
            if n.start < new_end:
                tmpnotes.append(n)
        i.notes = tmpnotes
        idx += 1
    gen.write(savepath)
 

def accompany2(savepath, midipath):
    # cdt.set_melody(midipath)
    # cdt.set_meta(tonic=cdt.Key.C, meter='4/4')
    # cdt.set_segmentation('A4')
    # cdt.set_output_style(cdt.Style.POP_STANDARD)
    # cdt.generate_save(savepath)
    # # cdt.generate_save(savepath,texture_prefilter=(2,1))
    # # a tuple (a, b) controlling rhythmic patters. a,
    # # b can be integers in [0, 4], each controlling horrizontal rhythmic density and vertical voice number. Ther
    # # higher number, the denser rhythms.
    # print("Accompany Done!")
    # # beat_tap_wav_write_pretty_midi(savepath, tmp_path)
    import pypianoroll
    # condition track:[4,48,84,1]
    n_bar = 4
    beat_resolution = 12
    n_timesteps = n_bar * beat_resolution * 4  # 4/4 拍
    n_pitch = 128
    # pm = pretty_midi.PrettyMIDI(midi_file=midipath)
    # c = pm.get_piano_roll()
    pr = pypianoroll.Multitrack(beat_resolution=beat_resolution)
    pr.parse_midi(filepath=midipath, algorithm='custom')

    c = np.array(pr.tracks[0].pianoroll, dtype=int)
    effective_steps = c.shape[0]
    n_submidi = int(np.ceil(c.shape[0] / n_timesteps))  # 向上取整
    if c.shape[0] < n_timesteps:
        tmp = -np.ones([n_timesteps - c.shape[0], n_pitch],dtype=int)
        # tmp = -np.ones([n_timesteps - c.shape[0], n_pitch])
        pr.tracks[0].pianoroll = np.append(c, tmp, axis=0)
    else:
        # 不足补0
        tmp = -np.ones([n_submidi * n_timesteps - c.shape[0], n_pitch], dtype=int)
        # tmp = -np.ones([n_submidi * n_timesteps - c.shape[0], n_pitch])
        pr.tracks[0].pianoroll = np.append(c, tmp, axis=0)
    pr.write('tmp1.mid')
    num_of_a4 = n_submidi
    cdt.set_melody('tmp1.mid')
    cdt.set_meta(tonic=cdt.Key.C, meter='4/4')
    cdt.set_segmentation('A4'*num_of_a4)
    cdt.set_output_style(cdt.Style.POP_STANDARD)
    cdt.generate_save(savepath)
    # cdt.generate_save(savepath,texture_prefilter=(2,1))
    # a tuple (a, b) controlling rhythmic patters. a,
    # b can be integers in [0, 4], each controlling horrizontal rhythmic density and vertical voice number. Ther
    # higher number, the denser rhythms.
    print("Accompany Done!")
    # beat_tap_wav_write_pretty_midi(savepath, tmp_path)


def main():
    # duet('./test_midi/lala4-2.mid', midipath='./test_midi/lala4-1.mid')
    # beat_tap_wav_write_pretty_midi('./1.mid', './tmp.mid')
    accompany('./1111.mid','./tmp2.mid')

if __name__ == "__main__":
    main()
