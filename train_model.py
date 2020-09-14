from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
import os
import data_process


def create_model():
    input = Input(shape=(256, 256, 1,))
    encoder1 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(input)
    encoder2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder1)
    encoder3 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder2)
    encoder4 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder3)
    encoder5 = Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(encoder4)

    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder5)

    decoder1 = Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(encoder_output)
    merge_decoder1 = concatenate([encoder5, decoder1])
    decoder2 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2)(merge_decoder1)
    merge_decoder2 = concatenate([encoder4, decoder2])
    decoder3 = Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2)(merge_decoder2)
    merge_decoder3 = concatenate([encoder3, decoder3])
    decoder4 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2)(merge_decoder3)
    merge_decoder4 = concatenate([encoder2, decoder4])
    decoder5 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2)(merge_decoder4)
    merge_decoder5 = concatenate([encoder1, decoder5])
    output = Conv2DTranspose(2, (3, 3), activation='tanh', padding='same', strides=2)(merge_decoder5)
    model = Model(inputs=input, outputs=output)
    model.summary()
    return model


def train(nb_epochs, batch_size, learning_rate, save_path=os.getcwd(), split_data=True):
    model = create_model()
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['acc'])

    train_path = os.path.join(os.getcwd(), "training")
    val_path = os.path.join(os.getcwd(), "validation")
    train_images_path = os.path.join(train_path, 'images')
    val_images_path = os.path.join(val_path, 'images')
    nb_train = data_process.get_number_of_data(train_images_path)
    nb_val = data_process.get_number_of_data(val_images_path)
    print('Broj podataka za treniranje prije podjele:', nb_train)
    print('Broj podataka za testiranje prije podjele:', nb_val)
    if split_data:
        data_process.split_data(train_images_path, val_images_path, 0.1)
        nb_train = data_process.get_number_of_data(train_images_path)
        nb_val = data_process.get_number_of_data(val_images_path)
        print('Broj podataka za treniranje nakon podjele:', nb_train)
        print('Broj podataka za testiranje nakon podjele', nb_val)
    train_generator = data_process.generator(2, train_path)
    validation_generator = data_process.generator(2, val_path)

    train_steps = data_process.get_number_of_data(train_images_path) // batch_size
    val_steps = data_process.get_number_of_data(val_images_path) // batch_size

    model.fit(train_generator, batch_size=batch_size, steps_per_epoch=train_steps, epochs=nb_epochs,
              validation_data=validation_generator, validation_steps=val_steps)
    save_path = os.path.join(save_path, 'model.h5')
    model.save(save_path)


if __name__ == "__main__":
    batch_size = 32
    learning_rate = 1e-3
    nb_epochs = 30
    train(nb_epochs, batch_size, learning_rate)
