function layers = creer_lenet(tailleImage)
% Réseau de référence (essai E0) : LeNet-5 adapté à Food-11, avec des images couleur et 11 classes
% Écrit comme le réseau de CNN_1_layer_for_CIFAR10.m, mais avec 2 blocs
% convolution + pooling et 3 couches entièrement connectées
% La couche softmax finale donne les probabilités des classes
% Test : analyzeNetwork(creer_lenet([64 64]))   -> 338 571 paramètres

layers = [
    imageInputLayer([tailleImage 3], Name="input")             % 64x64x3
    convolution2dLayer(5, 6, Name="conv1")                     % 6 filtres 5x5   -> 60x60x6
    reluLayer(Name="relu1")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling1")         % -> 30x30x6
    convolution2dLayer(5, 16, Name="conv2")                    % 16 filtres 5x5  -> 26x26x16
    reluLayer(Name="relu2")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling2")         % -> 13x13x16
    fullyConnectedLayer(120, Name="fullyconnected1")           % -> 120
    reluLayer(Name="relu3")
    fullyConnectedLayer(84, Name="fullyconnected2")            % -> 84
    reluLayer(Name="relu4")
    fullyConnectedLayer(11, Name="fullyconnected3")            % 11 classes de Food-11
    softmaxLayer(Name="softmax")];
end
