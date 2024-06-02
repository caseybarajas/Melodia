import os
import pickle
import numpy as np
import warnings
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# Suppress the copyright warning thingie
warnings.filterwarnings("ignore", category=UserWarning)

# Define a generator for data loading
def data_generator(batch_size):
    while True:
        for i in range(0, len(all_notes) - sequence_length, batch_size):
            network_input = []
            network_output = []
            for j in range(i, min(i + batch_size, len(all_notes) - sequence_length)):
                sequence_in = all_notes[j:j + sequence_length]
                sequence_out = all_notes[j + sequence_length]
                network_input.append([note_to_int[char] for char in sequence_in])
                network_output.append(note_to_int[sequence_out])
            network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
            network_input = network_input / float(n_vocab)
            network_output = to_categorical(network_output)
            yield network_input, network_output

# Parse MIDI files and extract notes
bass_notes = []
treble_notes = []
midi_files = []
total_files = len(os.listdir('data'))
for i, file in enumerate(os.listdir('data')):
    if file.endswith(".mid") or file.endswith(".midi"):
        midi_files.append(file)
        midi = converter.parse(os.path.join('data', file))
        parts = instrument.partitionByInstrument(midi)
        if parts:  # file has instrument parts
            for part in parts.parts:
                if "Piano" in str(part):  # select piano parts
                    for event in part.recurse():
                        if isinstance(event, note.Note):
                            if event.pitch.octave < 4:
                                bass_notes.append(str(event.pitch))
                            else:
                                treble_notes.append(str(event.pitch))
                        elif isinstance(event, chord.Chord):
                            if event.root().octave < 4:
                                bass_notes.append('.'.join(str(n) for n in event.normalOrder))
                            else:
                                treble_notes.append('.'.join(str(n) for n in event.normalOrder))
        print("Finished parsing file", i+1, "out of", total_files, "files:", file)

print("MIDI files parsed:")
for file in midi_files:
    print(file)

# Concatenate bass_notes and treble_notes
all_notes = bass_notes + treble_notes

# Create a dictionary to map all_notes to integers
unique_notes = sorted(set(note for note in all_notes))
note_to_int = dict((note, number) for number, note in enumerate(unique_notes))

# Save the note_to_int dictionary as a pickle file
with open('note_to_int.pkl', 'wb') as f:
    pickle.dump(note_to_int, f)

# Create input sequences and the corresponding outputs
sequence_length = 100
network_input = []
network_output = []
for i in range(0, len(all_notes) - sequence_length, 1):
    sequence_in = all_notes[i:i + sequence_length]
    sequence_out = all_notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)
n_vocab = len(unique_notes)

# Save the n_vocab as a pickle file
with open('n_vocab.pkl', 'wb') as f:
    pickle.dump(n_vocab, f)

# Reshape the input into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# Normalize input
network_input = network_input / float(n_vocab)

# Save the network_input as a pickle file
with open('network_input.pkl', 'wb') as f:
    pickle.dump(network_input, f)

# One hot encode the output
network_output = to_categorical(network_output)

# Create the model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the checkpoint
filepath = "models/Melodia-{epoch:02d}-{loss:.4f}.keras"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Use a more efficient optimizer
model.compile(loss='categorical_crossentropy', optimizer='Nadam')

# Use early stopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
callbacks_list.append(early_stopping)

# Fit the model using the generator
model.fit(data_generator(batch_size=64), epochs=50, steps_per_epoch=len(all_notes) // 64, callbacks=callbacks_list)
