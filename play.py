import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

# Load the note-to-integer mapping
with open('note_to_int.pkl', 'rb') as f:
    note_to_int = pickle.load(f)

# Load the network_input from a pickle file
with open('network_input.pkl', 'rb') as f:
    network_input = pickle.load(f)
    
# Load the n_vocab from a pickle file
with open('n_vocab.pkl', 'rb') as f:
    n_vocab = pickle.load(f)

# Create a reverse mapping to decode the output of the model
int_to_note = dict((number, note) for number, note in enumerate(note_to_int))

# Load the trained model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

model.load_weights('#path_to_model_weights.keras#')

# Generate a sequence of notes
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []

for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)

    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)

    pattern = np.append(pattern,index)
    pattern = pattern[1:len(pattern)]

# Convert the output from integers to notes and create a MIDI file
offset = 0
output_notes = []

for pattern in prediction_output:
    # pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # pattern is a note
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # Increase offset each iteration so that notes do not stack
    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write('midi', fp='test_output.mid')