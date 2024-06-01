import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# Parse MIDI files and extract notes
bass_notes = []
treble_notes = []
for file in os.listdir('examples/'):
    if file.endswith(".mid") or file.endswith(".midi"):
        midi = converter.parse(os.path.join('examples/', file))
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

# Fit the model
model.fit(network_input, network_output, epochs=20, batch_size=64, callbacks=callbacks_list)