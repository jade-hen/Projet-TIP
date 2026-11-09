function layers = creer_lenet_3blocs(tailleImage)
% Variante pour l'essai E3 : LeNet avec batch norm (comme creer_lenet_bn.m)
% et un 3e bloc convolution + batch norm + ReLU + max pooling (32 filtres 5x5)
% Du coup, ce qui entre dans la première couche FC passe de 13x13x16 à 4x4x32
% Test : analyzeNetwork(creer_lenet_3blocs([64 64]))   -> 88 471 paramètres

layers = [
    imageInputLayer([tailleImage 3], Name="input")             % 64x64x3
    convolution2dLayer(5, 6, Name="conv1")                     % 6 filtres 5x5   -> 60x60x6
    batchNormalizationLayer(Name="batchnorm1")
    reluLayer(Name="relu1")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling1")         % -> 30x30x6
    convolution2dLayer(5, 16, Name="conv2")                    % 16 filtres 5x5  -> 26x26x16
    batchNormalizationLayer(Name="batchnorm2")
    reluLayer(Name="relu2")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling2")         % -> 13x13x16
    convolution2dLayer(5, 32, Name="conv3")                    % ← ajouté : 32 filtres 5x5 -> 9x9x32
    batchNormalizationLayer(Name="batchnorm3")                 % ← ajouté
    reluLayer(Name="relu3")                                    % ← ajouté
    maxPooling2dLayer(2, Stride=2, Name="maxpooling3")         % ← ajouté : -> 4x4x32
    fullyConnectedLayer(120, Name="fullyconnected1")           % 4x4x32 = 512 valeurs -> 120
    reluLayer(Name="relu4")
    fullyConnectedLayer(84, Name="fullyconnected2")            % -> 84
    reluLayer(Name="relu5")
    fullyConnectedLayer(11, Name="fullyconnected3")            % 11 classes de Food-11
    softmaxLayer(Name="softmax")];
end
