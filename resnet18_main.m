%% Étape 2 : chargement des données
dataLocation = fullfile('food-11');
imageSize = [224 224];                          % taille d'entrée de ResNet-18
trainPart = 0.85;                               % 85 % train / 15 % validation

rng(0);   % même découpage à chaque fois


if isfile('food11_224.mat')
    fprintf('Chargement des images depuis la sauvegarde précédente\n');
    load('food11_224.mat');
    fprintf('Récupération des données terminée\n');
else
    fprintf('Chargement des images depuis le disque\n');
    [XTrain, TTrain, XValid, TValid, XTest, testNames] = load_and_format_data(dataLocation, imageSize, trainPart);
    % Sauvegarde pour ne pas recharger à chaque fois
    save('food11_224.mat', 'XTrain', 'TTrain', 'XValid', 'TValid', 'XTest', 'testNames', '-v7.3');
end

% Vérifications
size(XTrain)
size(XValid)
class(XTrain)
countcats(TTrain)    % nombre d'images par classe

% Afficher quelques images pour vérifier
figure; idx = randperm(size(XTrain,4), 8);
for k = 1:8
    subplot(2,4,k); imshow(XTrain(:,:,:,idx(k))); title(string(TTrain(idx(k))));
end



%% Étape 3 : construction du réseau
classNames = categories(TTrain);
numClasses = numel(classNames);   % 11

% ResNet-18 pré-entraîné sur ImageNet, dernière couche déjà remplacée par 11 sorties
net = imagePretrainedNetwork("resnet18", NumClasses=numClasses);

% Nom de la dernière couche entièrement connectée (la nouvelle, à apprendre plus vite)
isFC   = arrayfun(@(L) isa(L, 'nnet.cnn.layer.FullyConnectedLayer'), net.Layers);
fcName = net.Layers(find(isFC, 1, 'last')).Name
net = setLearnRateFactor(net, fcName, "Weights", 10);
net = setLearnRateFactor(net, fcName, "Bias", 10);

%% Test de vitesse : 1 époque sur 500 images
idx = randperm(size(XTrain,4), 500);
optionsTest = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=1, ...
    MiniBatchSize=32, ...
    ExecutionEnvironment="cpu", ...
    Plots="training-progress", ...
    OutputFcn=makeCourseProgressOutputFcn(500, 32, 1, 10), ...
    Metrics="accuracy", ...
    ValidationData={XValid, TValid}, ...
    Verbose=true, VerboseFrequency=4);



tic;
netTest = trainnet(XTrain(:,:,:,idx), TTrain(idx), net, "crossentropy", optionsTest);
t = toc;
fprintf('500 images : %.0f s  ->  estimation 1 époque complète : %.0f min\n', ...
        t, t / 500 * size(XTrain,4) / 60);



confusion_matrix(netTest, XValid, TValid)


export_results_test(netTest, XTest, testNames, classNames)