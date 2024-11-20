import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Load the preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    note_to_int = data['note_to_int']
    network_input = data['network_input']
    n_vocab = data['n_vocab']

# Create a reverse mapping to decode the output of the model
int_to_note = dict((number, note) for note, number in note_to_int.items())

# Load the trained model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

model.load_weights('models/melodia_final_model.keras')

# Generate a sequence of notes
start = np.random.randint(0, len(network_input)-1)
pattern = network_input[start]
prediction_output = []

# Generate notes
for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)
    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern = np.append(pattern[1:], index)

# Convert the output to MIDI
offset = 0
output_notes = []
for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        chord_notes = pattern.split('.')
        notes = []
        for current_note in chord_notes:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.storedInstrument = instrument.Piano()
        new_note.offset = offset
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='tests/generated_output.mid')