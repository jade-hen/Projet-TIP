clear; clc; close all;

%% 1. Chargement du modele pré-entrainé
numClasses = 11;
modelName  = "mobilenetv2";
net = imagePretrainedNetwork(modelName, NumClasses=numClasses);

% taille d'entrée attendue par le réseau (224x224 pour mobilenetv2)
inputSize = net.Layers(1).InputSize;
imageSize = inputSize(1:2);

%% 2. Chargement et formatage des données
dataLocation = "food-11";   % contient les sous-dossiers "train" et "test"
trainPart    = 0.8;         % 80 % entrainement, 20 % validation
rng(42);                     % même découpage à chaque fois

% le nom du cache dépend de la taille ET du découpage
cacheFile = "food11_" + imageSize(1) + "_" + round(trainPart*100) + ".mat";
if isfile(cacheFile)
    load(cacheFile, "XTrain", "TTrain", "XValid", "TValid", "XTest", "testNames");
else
    [XTrain, TTrain, XValid, TValid, XTest, testNames] = ...
        load_and_format_data(dataLocation, imageSize, trainPart);
    save(cacheFile, "XTrain", "TTrain", "XValid", "TValid", "XTest", "testNames", "-v7.3");
end
classNames = categories(TTrain);

%% 3. Options d'entraînement
miniBatchSize = 16;
itersPerEpoch = floor(size(XTrain,4) / miniBatchSize);

options = trainingOptions("adam", ...
    InitialLearnRate=1e-3, ...
    MaxEpochs=5, ...                         
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData={XValid, TValid}, ...
    ValidationFrequency=itersPerEpoch, ...   
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=true, VerboseFrequency=50);

%% 4. Entraînement (images en tableau 4D + étiquettes catégorielles)
trainedNet = trainnet(XTrain, TTrain, net, "crossentropy", options);
fprintf("Nombre de paramètres : %.2f M\n", sum(cellfun(@numel, trainedNet.Learnables.Value)) / 1e6);
%% 5. Evaluation sur la validation
scoresVal = minibatchpredict(trainedNet, XValid);
YVal = scores2label(scoresVal, classNames);
accuracy = mean(YVal == TValid);
fprintf("Precision sur la validation : %.2f %%\n", accuracy*100);

figure
confusionchart(TValid, YVal);
title("Matrice de confusion (validation)");

%% 6. Predictions sur le jeu de test + export JSON
% (les predictions sont calculees a l'interieur de export_results_test)
export_results_test(trainedNet, XTest, testNames, classNames);

% apercu de quelques images de test avec leur prediction
YTest = scores2label(minibatchpredict(trainedNet, XTest), classNames);
figure
idx = randperm(size(XTest, 4), 9);
for k = 1:9
    subplot(3, 3, k)
    imshow(XTest(:,:,:,idx(k)))
    title(string(YTest(idx(k))))
end

%% 7. Sauvegarde
save("food11_trainedNet.mat", "trainedNet", "classNames");