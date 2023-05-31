import numpy as np
from pypianoroll import Multitrack, Track 
import pypianoroll as pypiano
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ipdb
import os
from main import creat_directory, clean_folder
import pypianoroll


def find_pitch(song,volume=40):   # song shape(128,128), which is (time step, pitch)
    for time in range(song.shape[0]):
        step = song[time,:]
        max_index = np.argmax(step)
        for i in range(len(step)):
            if i ==max_index:
                song[time,i] = volume
            else:
                song[time,i] = 0
    return song

def reshape_bar(song):
    eight_bar = song[0]
    for i in range(7):
        b = song[i+1]
        eight_bar  = np.concatenate((eight_bar,b),axis=0)
    eight_bar = eight_bar.astype(float)
    print("A bar's shape: {}".format(eight_bar.shape))
    return eight_bar

def make_a_track(eight_bar_binarized,track_name ='melody' ,instrument=0):
    track = pypiano.StandardTrack(pianoroll=eight_bar_binarized, program=instrument, is_drum=False,name=track_name)
    return track

def make_a_demo(track1,track2,song_idx):
    sample_name = 'sample_'+str(song_idx)
    tempo = np.full(768, 120, dtype=float)
    multitrack = Multitrack(tracks=[track1,track2], tempo=tempo,resolution=4)
    return multitrack


def chord_list(chord,idx):

    one_song_chord = chord[idx]
    song_chord = []
    for i in range(len(one_song_chord)):
        bar_idx = []
        one_bar_chord = one_song_chord[i]
        bar_idx.append(int(one_bar_chord[0][12]))
        max_idx = np.argmax(one_bar_chord[:11])
        bar_idx.append(max_idx)
        song_chord.append(bar_idx)
    return song_chord


def build_chord_map():
    c_maj  = [60,64,67]
    c_min  = [60,63,67]
    chord_map = []
    chord_list_maj = []
    chord_list_min = []
    chord_list_maj.append(c_maj)
    chord_list_min.append(c_min)
    for i in range(11):
        chord = [x+1 for x in c_maj] 
        c_maj = chord
        chord_list_maj.append(chord)
        chord_min = [x+1 for x in c_min]
        chord_list_min.append(chord_min)
        c_min = chord_min
    chord_map.append(chord_list_maj)
    chord_list_min[:] = chord_list_min[9:] + chord_list_min[0:9]
    chord_map.append(chord_list_min)
    return chord_map

def decode_chord(maj_min,which_chord):

    chord_map = build_chord_map()
    chord = chord_map[maj_min][which_chord]

    return chord

def get_chord(song_chord):
    chord_player = []
    for item in song_chord:
        maj_min = item[0]
        which_chord = item[1]
        answer_chord = decode_chord(maj_min,which_chord)
        chord_player.append(answer_chord)
    return chord_player

def make_chord_track(chord,instrument,volume=40):
    pianoroll = np.zeros((128, 128))
    for i in range(len(chord)):
        st = 16*i
        ed = st + 16
        chord_pitch = chord[i]
        pianoroll[st:ed, chord_pitch] = volume
    track = pypiano.StandardTrack(pianoroll=pianoroll, program=instrument, is_drum=False,
                  name='chord')
    return track



def main():
    model_ver = int(input("Model: "))
    attempt = int(input("Attempt: "))
    is_augmented = int(input("augmented: ")) 
    aug_path = "augmented"
    if not is_augmented:
        aug_path = "not_augmented"

    init_path = f"./{model_ver}/{aug_path}/{model_ver}.{attempt}"
    output_path = init_path + "/output"
    data = np.load(os.path.join(output_path, 'output_songs.npy'))  # m, 1, 16, 128
    chord = np.load(os.path.join(output_path, 'output_chords.npy')) # m, 13
    # instrument = input('which instrument you want to play? from 0 to 128,default=0:')
    # volume     = input('how loud you want to play? from 1 to 127,default= 40:')
    instrument = 0
    volume = 40
    sample_name = "test_song"
    save_path = init_path + "/demo"
    creat_directory(save_path)
    clean_folder(save_path)
    for i in range(len(data)):
        if i % 20 == 0:
            one_song = data[i]
            song = []
            for item in one_song:
                item = item.reshape(16,128)
                song.append(item)
            eight_bar = reshape_bar(song)
            eight_bar_binarized = find_pitch(eight_bar,volume)
            track = make_a_track(eight_bar_binarized,instrument=instrument)
            
            song_chord = chord_list(chord,i)
            chord_player = get_chord(song_chord)
            np.save(os.path.join(save_path, f'chord_{i}.npy'),chord_player)
            chord_track = make_chord_track(chord_player,instrument,volume)
            multitrack = make_a_demo(track,chord_track,i)
            multitrack.plot()
            plt.savefig(os.path.join(save_path, f"{sample_name}_{i}.png"))
            pypianoroll.write(os.path.join(save_path, f"{sample_name}_{i}.mid"), multitrack=multitrack)
            print(str(sample_name)+'saved')
            os.remove(os.path.join(save_path, f'chord_{i}.npy'))




if __name__ == "__main__" :

    main()









