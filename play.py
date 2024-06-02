import os
import pickle
import numpy as np
import datetime
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

model.load_weights('models/#')

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

# Create two separate streams for the bass and treble parts
bass_stream = stream.Part()
treble_stream = stream.Part()

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

        # Add the note to the bass or treble stream based on its pitch
        if new_note.pitch.midi < 60:
            bass_stream.append(new_note)
        else:
            treble_stream.append(new_note)

    # pattern is a note
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()

        # Add the note to the bass or treble stream based on its pitch
        if new_note.pitch.midi < 60:
            bass_stream.append(new_note)
        else:
            treble_stream.append(new_note)
        
    # Increase offset each iteration so that notes do not stack
    offset += 0.5

# Combine the bass and treble streams into a single stream
midi_stream = stream.Stream([bass_stream, treble_stream])

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_file = f"tests/test_output_{current_time}.mid"
midi_stream.write('midi', fp=output_file)

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_file = f"tests/test_output_{current_time}.mid"
midi_stream.write('midi', fp=output_file)