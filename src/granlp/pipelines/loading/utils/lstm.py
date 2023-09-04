from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


def model_add():
    inputs = Input(shape=(100, ))
    x = Embedding(20000, 128)(inputs)
    x = Bidirectional(LSTM(50))(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
model = model_add()
print(model.summary()
    

    

    x = concatenate([avg_pool1, max_pool1, avg_pool2, max_pool2])



    x = Dense(6, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)

    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])

    history = model.fit(X_train, Y_train, batch_size = 128, epochs = 3, validation_data = (X_valid, Y_valid), 

                        verbose = 1, callbacks = [ra_val, check_point, early_stop])

    model = load_model(file_path)

    return model
