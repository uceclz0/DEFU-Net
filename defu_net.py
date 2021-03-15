"""Import relative package"""
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, concatenate, BatchNormalization, \
    LeakyReLU, MaxPooling2D, Add, add, UpSampling2D

# DEFU_Net


def incep_block(input_layer, nb_filters):
    """Inception Block with dialtion;
    nb_filters: the input number of feature maps"""
    branch_a = Conv2D(nb_filters, 1,
                      activation='relu', strides=2)(input_layer)
    branch_b = Conv2D(nb_filters, 1, activation='relu')(input_layer)
    branch_b = Conv2D(nb_filters, 3, activation='relu',
                      padding='same', strides=2)(branch_b)

    branch_c = AveragePooling2D(3, strides=2, padding='same')(input_layer)
    branch_c = Conv2D(nb_filters, 3, activation='relu',
                      padding='same', dilation_rate=(1, 2))(branch_c)

    branch_d = Conv2D(nb_filters, 1, activation='relu')(input_layer)
    branch_d = Conv2D(nb_filters, 3, activation='relu',
                      padding='same', dilation_rate=(3, 1))(branch_d)
    branch_d = Conv2D(nb_filters, 3, activation='relu',
                      padding='same', strides=2)(branch_d)

    # Concatenate all the branches
    output = concatenate(
        [branch_a, branch_b, branch_c, branch_d], axis=-1)
    return output

# DCRC (Densely Connected Recurrent Convolution)


def rec_des_block(input_layer, out_n_filters, batch_normalization=True, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='channels_last'):
    """DCRC Block;
    input_layer: input;
    out_n_filters: the output number of feature maps"""
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    dense = []
    layer = skip_layer
    for j in range(2):
        # skip = []
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = LeakyReLU()(layer1)
                # skip.append(layer1)
            # if i == 1:
            #     layer = skip[0]
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = LeakyReLU()(layer1)

        layer = concatenate([layer1, skip_layer], axis=3)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            layer)
        layer = LeakyReLU()(layer)
        dense.append(layer)

    out_layer = concatenate([dense[1], dense[0]], axis=3)
    out_layer = concatenate([out_layer, skip_layer], axis=3)
    out_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
        out_layer)
    out_layer = LeakyReLU()(out_layer)
    return out_layer


def incep_des_encoder(input_layer):
    """encoding part of DEFU_Net"""
    to_decoder = []

    main_path1 = rec_des_block(input_layer, 64)
    main1 = MaxPooling2D((2, 2), data_format='channels_last')(main_path1)
    to_decoder.append(main_path1)

    main_path2 = rec_des_block(main1, 128)
    m_2 = MaxPooling2D((2, 2), data_format='channels_last')(main_path2)
    fcn1 = incep_block(main_path1, 128//4)
    # main2 = concatenate([main_path2, fcn1])
    main2 = Add()([main_path2, fcn1])
    to_decoder.append(main2)

    main_path3 = rec_des_block(m_2, 256)
    m_3 = MaxPooling2D((2, 2), data_format='channels_last')(main_path3)
    fcn2 = incep_block(main2, 256//4)
    # main3 = concatenate([main_path3, fcn2])
    main3 = Add()([main_path3, fcn2])
    to_decoder.append(main3)

    main_path4 = rec_des_block(m_3, 512)
    m_4 = MaxPooling2D((2, 2), data_format='channels_last')(main_path4)
    fcn3 = incep_block(main3, 512//4)
    main4 = Add()([main_path4, fcn3])
    to_decoder.append(main4)

    return to_decoder, m_4


def incep_des_decoder(input_layer, from_encoder):
    """Decoding part of DEFU_Net"""
    main_path = UpSampling2D(size=(2, 2))(input_layer)
    main_path = concatenate([main_path, from_encoder[3]], axis=3)
    main_path = rec_des_block(main_path, 256)

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = rec_des_block(main_path, 128)

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = rec_des_block(main_path, 64)

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = rec_des_block(main_path, 64)

    return main_path


def dense_r_incep_unet(input_size):
    """DEFU_Net"""
    inputs = Input(shape=input_size)

    to_decoder, m_4 = incep_des_encoder(inputs)
    fcn4 = incep_block(to_decoder[3], 512//4)

    path = m_4

    bottom = Add()([path, fcn4])
    path = rec_des_block(bottom, 512)

    path = incep_des_decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(path)

    return Model(input=inputs, output=path)
