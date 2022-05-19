import copy
import os
import random

import note_seq
import numpy as np
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2

import chorderator as cdt

# PROJECT_DIR = os.path.dirname(os.getcwd())
PROJECT_DIR = r'D:\MyFiles\FinalTest\music'
DATA_DIR = os.path.join(PROJECT_DIR, 'resources')
DATA_MIDI = os.path.join(DATA_DIR, 'midi', 'in')
DATA_PIANOROLL = os.path.join(DATA_DIR, 'pianorolls')
SOUND_FONT_PATH = os.path.join(DATA_DIR, 'soundfont')

MUSIC_RNN_TYPE = 'attention_rnn'
MUSIC_RNN = os.path.join(DATA_DIR, 'model_checkpoints', 'music_rnn', MUSIC_RNN_TYPE + '.mag')

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

def change_tempo(note_sequence, new_tempo):
    new_sequence = copy.deepcopy(note_sequence)
    ratio = note_sequence.tempos[0].qpm / new_tempo
    for note in new_sequence.notes:
        note.start_time = note.start_time * ratio
        note.end_time = note.end_time * ratio
    new_sequence.tempos[0].qpm = new_tempo
    return new_sequence

def duet(savepath, midipath, temperature=0.8, ):
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
    print('?? Done!')

    input_sequence = note_seq.midi_file_to_note_sequence(midipath)

    # Set the start time to begin on the next step after the last note ends.
    qpm = input_sequence.tempos[0].qpm + 20
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


def accompany_result_to_midi(savepath, npy, effective_steps):
    import pypianoroll
    velocity = [20, 127, 10, 10, 10]
    programs = [127, 0, 46, 33, 0]  # [0, 0, 25, 33, 48] # drum,piano,harp/guitar,bass,string
    is_drums = [True, False, False, False, False]  # [1, 0, 0, 0, 0]
    tracks = []
    a = npy
    # timestep*pitch
    for i in range(5):
        b = a[..., i]
        b = b.reshape(-1, 84)[:effective_steps]
        b[b > 0] = velocity[i]
        b = np.pad(b, ((0, 0), (24, 20)), constant_values=-1)
        track = pypianoroll.Track(b, program=programs[i], is_drum=is_drums[i])
        tracks.append(track)

    pr = pypianoroll.Multitrack(tracks=tracks, tempo=120, beat_resolution=12)
    pr.write('tmp.mid')
    # pm = beat_tap(pr.to_pretty_midi())
    # pm.write(savepath)
    beat_tap_wav_write_pretty_midi(savepath,'tmp.mid')


def accompany(savepath, midipath):
    import pypianoroll
    # condition track:[4,48,84,1]
    n_bar = 4
    beat_resolution = 12
    n_timesteps = n_bar * beat_resolution * 4  # 4/4 拍
    n_pitch = 84
    pr = pypianoroll.Multitrack(beat_resolution=beat_resolution)
    pr.parse_midi(filepath=midipath, algorithm='custom')
    c = np.array(pr.tracks[0].pianoroll, dtype=int)[:, 24:108]
    # change to -1,1 as input of musegan
    c[c == c.min()] = -1
    c[c == c.max()] = 1
    effective_steps = c.shape[0]
    # 204*84
    if c.shape[0] < n_timesteps:
        tmp = -np.ones([n_timesteps - c.shape[0], n_pitch], dtype=int)
        m = np.append(c, tmp, axis=0).reshape(n_bar, 4 * beat_resolution, n_pitch, -1)  # 4,48,84,1
        result = musegan(m, RESULT_DIR, str('0'))[0]  # 4,48,84,5
    else:
        n_submidi = int(np.ceil(c.shape[0] / n_timesteps))  # 向上取整
        # 不足补0
        tmp = -np.ones([n_submidi * n_timesteps - c.shape[0], n_pitch], dtype=int)
        c = np.append(c, tmp, axis=0)
        result = []
        for i in range(n_submidi):
            cc = c[i * n_timesteps:(i + 1) * n_timesteps, :].reshape(n_bar, 4 * beat_resolution, n_pitch, -1)
            result.append(musegan(cc, RESULT_DIR, str(i))[0])  # musegan返回[1,4,48,84,1]的
        result = np.array(result)
    accompany_result_to_midi(savepath, result, effective_steps)
    print("Accompany Done!")


def beat_tap(pm):
    import pretty_midi
    downbeats = pm.get_downbeats()
    tap = pretty_midi.Instrument(program=112)

    for t in downbeats:
        new_note = pretty_midi.Note(velocity=60, pitch=84, start=t, end=(t + 0.5))
        tap.notes.append(new_note)
    pm.instruments.append(tap)
    print("Beat Tracking Done!")
    return pm

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


def generate_chords(key, sType="major"):
    chromatic_scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def getScale(key=key, sType=sType):
        if sType == "major":
            formula = [2, 2, 1, 2, 2]
        else:
            formula = [2, 1, 2, 2, 1]

        scale = []
        scale.append(key)
        for i in range(len(chromatic_scale)):
            if chromatic_scale[i] == key:
                break
        j = 0
        while len(scale) < 6:
            if i + formula[j] < 12:
                i += formula[j]
                j += 1
            else:
                i = 12 - (i + formula[j])
                j += 1
            scale.append(chromatic_scale[i])
        return scale

    # key = random.choice(chromatic_scale)
    key = key
    toss = [0, 1]
    j = random.choice(toss)
    if j == 1:
        sType = "minor"
    else:
        sType = "major"

    scale = getScale(key, sType)

    no_of_chords = random.randrange(3, 6)

    chords = ''
    picked = []
    while chords.count(' ') < no_of_chords:
        k = random.randrange(6)
        if k not in picked:
            picked.append(k)
            if k in [1, 2, 5]:
                chords += scale[k] + 'm' + ' '
            else:
                chords += scale[k] + ' '

    return chords


def add_chords(savepath, midipath):
    import pretty_midi
    from pychord import Chord
    chords_str = generate_chords(key='C', sType='major').strip().split(" ")
    chords = [Chord(c) for c in chords_str]
    midi_data = pretty_midi.PrettyMIDI(midipath)
    # origin_time = midi_data.get_onsets()
    all_notes = midi_data.instruments[0].notes
    start = all_notes[0].start
    for note in all_notes:
        note.start -= start
        note.end -= start
    # end_time = np.array([x - start for x in origin_time])
    # midi_data.adjust_times(origin_time, end_time)
    downbeats = midi_data.get_downbeats()  # downbeats处加chord
    chords = chords[:len(downbeats)]
    chords_track = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program('Acoustic Grand Piano'))
    for t, chord in zip(downbeats, chords):
        for note_name in chord.components_with_pitch(root_pitch=4):
            note_number = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=t, end=t + 1)
            chords_track.notes.append(note)
    midi_data.instruments.append(chords_track)
    midi_data.write(savepath)
    print("Chords add done!")


def accompany(savepath, midipath):
    import pretty_midi
    import time
    dirname = 'D:\\MyFiles\\FinalTest\\music\\AI-music\\test_midi\\'
    name = 'memoria.mid'
    cdt.set_melody(dirname + name)
    pm = pretty_midi.PrettyMIDI(dirname + name)
    tic = time.time()
    cdt.set_meta(tonic=cdt.Key.C, meter='4/4')
    cdt.set_segmentation('A4')
    cdt.set_output_style(cdt.Style.POP_STANDARD)
    cdt.generate_save(name)
    toc = time.time()
    print(toc - tic)

    tmp_path = './tmp.mid'
    add_chords(savepath=tmp_path, midipath=midipath)
    beat_tap_wav_write_pretty_midi(savepath, tmp_path)

def main():
    # duet('./test_midi/lala4-2.mid', midipath='./test_midi/lala4-1.mid')
    # beat_tap_wav_write_pretty_midi('./1.mid', './tmp.mid')
    accompany('./final.mid','./1.mid')

if __name__ == "__main__":
    main()
