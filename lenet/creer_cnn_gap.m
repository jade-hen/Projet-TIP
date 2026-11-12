function layers = creer_cnn_gap(tailleImage)
% Version finale (essai E6), celle qu'utilise main_lenet.m : 4 blocs convolution 3x3 (avec padding) + batch norm
% + ReLU + max pooling, puis un global average pooling à la place des couches FC 120 et 84
% Test : analyzeNetwork(creer_cnn_gap([64 64]))   -> 99 339 paramètres

layers = [
    imageInputLayer([tailleImage 3], Name="input")                     % 64x64x3
    convolution2dLayer(3, 16, Padding="same", Name="conv1")            % 16 filtres 3x3 -> 64x64x16
    batchNormalizationLayer(Name="batchnorm1")
    reluLayer(Name="relu1")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling1")                 % -> 32x32x16
    convolution2dLayer(3, 32, Padding="same", Name="conv2")            % 32 filtres 3x3 -> 32x32x32
    batchNormalizationLayer(Name="batchnorm2")
    reluLayer(Name="relu2")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling2")                 % -> 16x16x32
    convolution2dLayer(3, 64, Padding="same", Name="conv3")            % 64 filtres 3x3 -> 16x16x64
    batchNormalizationLayer(Name="batchnorm3")
    reluLayer(Name="relu3")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling3")                 % -> 8x8x64
    convolution2dLayer(3, 128, Padding="same", Name="conv4")           % 128 filtres 3x3 -> 8x8x128
    batchNormalizationLayer(Name="batchnorm4")
    reluLayer(Name="relu4")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling4")                 % -> 4x4x128
    globalAveragePooling2dLayer(Name="gap")                            % moyenne de chaque carte -> 1x1x128
    fullyConnectedLayer(11, Name="fullyconnected")                     % 11 classes de Food-11
    softmaxLayer(Name="softmax")];
end
