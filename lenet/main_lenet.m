% main_lenet.m : notre réseau fait main (parti de LeNet-5) sur Food-11
% On utilise ici les 3 scripts du TD: CIFAR10_database.m, CNN_1_layer_for_CIFAR10.m et Confusion_matrix_CIFAR10.m
% Toutes les lignes qui ont été modifiées ou ajoutées par rapport au TP sont signalées par des flèches ← 


%% 1. Chargement des données (CIFAR10_database.m)
clearvars
close all

% Charger les images de Food-11 et les découper en entraînement / validation
tailleImage = [64 64];                                  % ← ajouté : taille des images en entrée du réseau
rng(42);                                                % ← ajouté : graine aléatoire 
[imgDataTrain, labelsTrain, imgDataVal, labelsVal] = ...
    load_and_format_data(fullfile('..', 'food-11', 'train'), tailleImage, 0.85);   % ← modifié : fonction de A, 85 % / 15 %
fprintf("Images : %d entraînement, %d validation\n", numel(labelsTrain), numel(labelsVal));   % ← ajouté pour le rapport
aug = imageDataAugmenter(RandXReflection=true, RandRotation=[-15 15], RandXTranslation=[-8 8], RandYTranslation=[-8 8]);   % ← ajouté (essai E4) : règles pour transformer les photos au hasard
augTrain = augmentedImageDatastore(tailleImage, imgDataTrain, labelsTrain, DataAugmentation=aug);                            % ← ajouté (essai E4) : les photos sont transformées à chaque passage

%% 2. Réseau et entraînement (CNN_1_layer_for_CIFAR10.m)
% Afficher 25 images d'entraînement au hasard
perm = randperm(numel(labelsTrain), 25);
subset = imgDataTrain(:, :, :, perm);
montage(subset)

% Le réseau utilisé est la version finale (essai E6), décrite dans creer_cnn_gap.m
% Les réseaux des essais précédents sont gardés dans le dossier essais_precedents
layers = creer_cnn_gap(tailleImage);                   % ← modifié : notre réseau au lieu du réseau à un bloc du TP

% Vérifier l'architecture
analyzeNetwork(layers)

% Réglages de l'entraînement. La validation est passée dans ValidationData pour suivre son accuracy et son erreur pendant l'entraînement
miniBatchSize = 500;
maxEpochs = 30;                                         % ← modifié : 30 au lieu de 16 (essai E5)
smoothingWindow = 10; % Lissage des courbes, en mini-batchs
progressFcn = makeCourseProgressOutputFcn( ...
    numel(labelsTrain), miniBatchSize, maxEpochs, smoothingWindow);

options = trainingOptions("adam", ...                  % ← modifié : adam au lieu de sgdm (essai E2)
    InitialLearnRate=0.001, ...
    MaxEpochs=maxEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData={imgDataVal, labelsVal}, ...         % ← modifié : la validation, jamais le test
    ValidationFrequency=50, ...
    Verbose=false, ...
    Plots="training-progress", ...
    OutputFcn=progressFcn, ...
    Metrics="accuracy");

net = trainnet(augTrain, layers, "crossentropy", options);   % ← modifié (essai E4) : on entraîne sur les photos transformées

% Évaluer le réseau sur l'entraînement et sur la validation
trainScores = minibatchpredict(net, imgDataTrain);
trainLabelsPred = scores2label(trainScores, categories(labelsTrain));
trainAccuracy = mean(trainLabelsPred == labelsTrain);
fprintf("Train-set accuracy: %.2f%%\n", 100 * trainAccuracy);

valScores = minibatchpredict(net, imgDataVal);                      % ← modifié : test → validation
valLabelsPred = scores2label(valScores, categories(labelsTrain));   % ← modifié : test → validation
valAccuracy = mean(valLabelsPred == labelsVal);                     % ← modifié : test → validation
fprintf("Validation-set accuracy: %.2f%%\n", 100 * valAccuracy);    % ← modifié : test → validation

%% 3. Matrice de confusion, F1 et frugalité (Confusion_matrix_CIFAR10.m)
scoresVal = minibatchpredict(net, imgDataVal);                      % ← modifié : test → validation
labelsPred = scores2label(scoresVal, categories(labelsTrain));      % ← modifié : test → validation

C = confusionmat(labelsVal, labelsPred);                            % ← modifié : test → validation
tp = diag(C);

predictedCounts = sum(C, 1)';
trueCounts = sum(C, 2);

precisionPerClass = tp ./ predictedCounts;
recallPerClass = tp ./ trueCounts;

% Classes jamais prédites ou absentes de la validation
precisionPerClass(predictedCounts == 0) = NaN;
recallPerClass(trueCounts == 0) = NaN;

macroPrecision = mean(precisionPerClass, 'omitnan');
macroRecall = mean(recallPerClass, 'omitnan');

% ← ajouté : F1 de chaque classe et F1 macro (la mesure du challenge)
f1PerClass = 2 * precisionPerClass .* recallPerClass ./ (precisionPerClass + recallPerClass);
f1PerClass(isnan(f1PerClass)) = 0;      % classe jamais prédite ou jamais trouvée : F1 = 0
macroF1 = mean(f1PerClass);

% ← ajouté : nombre de paramètres et score de frugalité (formule du sujet)
nbParams = sum(cellfun(@numel, net.Learnables.Value));
frugalite = macroF1 - 0.3 / (1 + 1/(nbParams/1e6));

figure
cm = confusionchart(labelsVal, labelsPred);                         % ← modifié : test → validation
cm.ColumnSummary = "column-normalized"; % Précision de chaque classe prédite
cm.RowSummary = "row-normalized";       % Rappel de chaque vraie classe
cm.Title = sprintf("Food-11 Confusion Matrix\nMacro precision: %.2f%% | Macro recall: %.2f%% | Macro F1: %.2f%%", ...
    100 * macroPrecision, 100 * macroRecall, 100 * macroF1);       % ← modifié : Food-11 et F1 macro

%% ← ajouté : résumé à recopier dans resultats_lenet.xlsx
fprintf("\nParamètres : %d | Acc. entraîn. : %.3f | Acc. valid. : %.3f | F1 macro : %.3f | Frugalité : %.3f\n", ...
    nbParams, trainAccuracy, valAccuracy, macroF1, frugalite);
disp(table(categories(labelsTrain), f1PerClass, VariableNames=["Classe", "F1"]))
