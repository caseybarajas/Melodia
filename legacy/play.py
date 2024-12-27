import os
import pickle
import numpy as np
import argparse
from music21 import converter, instrument, note, chord, stream, articulations
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

def parse_args():
    parser = argparse.ArgumentParser(description='Generate music using trained model')
    parser.add_argument('--length', type=int, default=500, help='Length of generated sequence')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--output', type=str, default='output.mid', help='Output file name')
    return parser.parse_args()

# Load the preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    note_to_int = data['note_to_int']
    duration_to_int = data['duration_to_int']
    instrument_to_int = data['instrument_to_int']
    articulation_to_int = data['articulation_to_int']
    network_input = data['network_input']
    n_vocab = data['n_vocab']

# Create reverse mappings
int_to_note = dict((number, note) for note, number in note_to_int.items())
int_to_duration = dict((number, duration) for duration, number in duration_to_int.items())
int_to_instrument = dict((number, inst) for inst, number in instrument_to_int.items())
int_to_articulation = dict((number, art) for art, number in articulation_to_int.items())

# Load the model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

model.load_weights('models/melodia_final_model.keras')

def generate_sequence(length, temperature=1.0):
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []
    
    for _ in range(length):
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
        prediction_input = prediction_input / float(n_vocab)
        prediction = model.predict(prediction_input, verbose=0)
        
        # Apply temperature scaling
        prediction = np.log(prediction) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        
        # Sample from the distribution
        probas = prediction[0]
        index = np.random.choice(len(probas), p=probas)
        
        # Get all features for this timestep
        note_val = int_to_note[index]
        duration_val = float(int_to_duration[pattern[0][1]])
        instrument_val = int_to_instrument[pattern[0][2]]
        articulation_val = int_to_articulation[pattern[0][3]]
        
        prediction_output.append({
            'note': note_val,
            'duration': duration_val,
            'instrument': instrument_val,
            'articulation': articulation_val
        })
        
        # Update pattern
        pattern = np.roll(pattern, -1, axis=0)
        pattern[-1] = [index, pattern[-1][1], pattern[-1][2], pattern[-1][3]]
    
    return prediction_output

def create_midi(prediction_output, output_file):
    offset = 0
    output_notes = []
    
    for pred in prediction_output:
        if ('.' in pred['note']) or pred['note'].isdigit():
            # Handle chord
            chord_notes = pred['note'].split('.')
            notes = []
            for current_note in chord_notes:
                new_note = note.Note(int(current_note))
                new_note.quarterLength = pred['duration']
                new_note.storedInstrument = instrument.Piano()  # Convert instrument string to object
                if pred['articulation'] != 'none':
                    art_class = getattr(articulations, pred['articulation'].split('.')[-1])
                    new_note.articulations.append(art_class())
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # Handle single note
            new_note = note.Note(pred['note'])
            new_note.quarterLength = pred['duration']
            new_note.storedInstrument = instrument.Piano()
            if pred['articulation'] != 'none':
                art_class = getattr(articulations, pred['articulation'].split('.')[-1])
                new_note.articulations.append(art_class())
            new_note.offset = offset
            output_notes.append(new_note)
        
        offset += pred['duration']

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

def main():
    args = parse_args()
    prediction_output = generate_sequence(args.length, args.temperature)
    create_midi(prediction_output, args.output)

if __name__ == '__main__':
    main()