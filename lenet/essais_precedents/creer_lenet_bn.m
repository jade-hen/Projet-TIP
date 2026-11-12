function layers = creer_lenet_bn(tailleImage)
% Variante de creer_lenet.m pour l'essai E1 : on ajoute une batch normalization après
% chaque convolution, comme dans le réseau du TP (conv -> batchnorm -> relu -> pool)
% Test : analyzeNetwork(creer_lenet_bn([64 64]))   -> 338 615 paramètres

layers = [
    imageInputLayer([tailleImage 3], Name="input")             % 64x64x3
    convolution2dLayer(5, 6, Name="conv1")                     % 6 filtres 5x5   -> 60x60x6
    batchNormalizationLayer(Name="batchnorm1")                 % ← ajouté (comme dans le TP)
    reluLayer(Name="relu1")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling1")         % -> 30x30x6
    convolution2dLayer(5, 16, Name="conv2")                    % 16 filtres 5x5  -> 26x26x16
    batchNormalizationLayer(Name="batchnorm2")                 % ← ajouté (comme dans le TP)
    reluLayer(Name="relu2")
    maxPooling2dLayer(2, Stride=2, Name="maxpooling2")         % -> 13x13x16
    fullyConnectedLayer(120, Name="fullyconnected1")           % -> 120
    reluLayer(Name="relu3")
    fullyConnectedLayer(84, Name="fullyconnected2")            % -> 84
    reluLayer(Name="relu4")
    fullyConnectedLayer(11, Name="fullyconnected3")            % 11 classes de Food-11
    softmaxLayer(Name="softmax")];
end
