import os
import pickle
import numpy as np
import warnings
from music21 import converter, instrument, note, chord, stream
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Suppress the warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_midi_files(data_dir):
    bass_notes = []
    treble_notes = []
    midi_files = []
    total_files = len([f for f in os.listdir(data_dir) if f.endswith(('.mid', '.midi'))])
    
    for i, file in enumerate(os.listdir(data_dir)):
        if file.endswith((".mid", ".midi")):
            midi_files.append(file)
            midi = converter.parse(os.path.join(data_dir, file))
            parts = instrument.partitionByInstrument(midi)
            if parts:
                for part in parts.parts:
                    if "Piano" in str(part):
                        for event in part.recurse():
                            if isinstance(event, note.Note):
                                (bass_notes if event.pitch.octave < 4 else treble_notes).append(str(event.pitch))
                            elif isinstance(event, chord.Chord):
                                chord_str = '.'.join(str(n) for n in event.normalOrder)
                                (bass_notes if event.root().octave < 4 else treble_notes).append(chord_str)
            print(f"Finished parsing file {i+1} out of {total_files} files: {file}")
    
    return bass_notes, treble_notes, midi_files

def prepare_sequences(notes, sequence_length=100):
    note_to_int = dict((note, number) for number, note in enumerate(sorted(set(notes))))
    
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    
    n_patterns = len(network_input)
    n_vocab = len(set(notes))
    
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)
    
    return network_input, network_output, n_vocab, note_to_int

def create_model(input_shape, n_vocab):
    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def main():
    data_dir = 'data'
    bass_notes, treble_notes, midi_files = parse_midi_files(data_dir)
    all_notes = bass_notes + treble_notes

    network_input, network_output, n_vocab, note_to_int = prepare_sequences(all_notes)

    # Save preprocessed data
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump({
            'note_to_int': note_to_int,
            'n_vocab': n_vocab,
            'network_input': network_input,
        }, f)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(network_input, network_output, test_size=0.2, random_state=42)

    model = create_model((X_train.shape[1], X_train.shape[2]), n_vocab)

    # Define callbacks
    filepath = "models/Melodia-{epoch:02d}-{val_loss:.4f}.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    
    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    # Fit the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks_list
    )

    # Save the final model
    model.save('melodia_final_model.keras')

    print("Training completed. Model saved as 'melodia_final_model.keras'")

if __name__ == "__main__":
    main()