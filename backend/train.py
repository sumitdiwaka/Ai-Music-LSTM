import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100  # How many notes to look at before predicting the next one

    # Get all unique pitch names
    pitchnames = sorted(list(set(notes)))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    # One-hot encode the output
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return (network_input, network_output)

def create_network(network_in, n_vocab):
    """ Create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        256, # Reduced from 512 for faster training
        input_shape=(network_in.shape[1], network_in.shape[2]),
        recurrent_dropout=0.2, # Reduced dropout
        return_sequences=True
    ))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(256))
    model.add(Dense(128)) # Reduced from 256
    model.add(Dropout(0.2))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train():
    """ Train a Neural Network to generate music """
    # Load the notes file
    notes_path = os.path.join('model', 'notes')
    with open(notes_path, 'rb') as filepath:
        notes = pickle.load(filepath)

    # --- CHANGE FOR FASTER TRAINING ---
    # Use only a subset of the data to speed up training
    notes = notes[:20000] # Using first 20,000 notes. Remove this line for full training.
    
    # Get the number of unique pitches
    n_vocab = len(set(notes))

    # Prepare the sequences for the model
    network_in, network_out = prepare_sequences(notes, n_vocab)

    # Create the model
    model = create_network(network_in, n_vocab)

    print("Starting model training... (Fast Mode)")

    # Define the checkpoint to save the best model weights
    weights_path = os.path.join('model', 'weights-best-fast.keras') # Use a different file for the fast model
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # Train the model
    model.fit(
        network_in, 
        network_out, 
        # --- CHANGE FOR FASTER TRAINING ---
        epochs=20, # Reduced from 100
        batch_size=128, # Increased batch size for potential speed up
        callbacks=callbacks_list
    )
    
    print("Model training finished.")

if __name__ == '__main__':
    train()
