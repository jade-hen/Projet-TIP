% resnet18_main.m : apprentissage par transfert de ResNet-18 sur Food-11
%
% Un seul script pour toutes les expériences ResNet-18 :
%   1. on choisit les réglages dans la section 0 (et UNIQUEMENT là),
%   2. on lance le script,
%   3. les scores s'ajoutent à resultats_resnet18.csv (tableau du rapport),
%      la matrice de confusion est enregistrée en PNG.
%
% Expériences prévues (étude d'ablation) :
%   E1  modeGel = "fc"    -> seule la dernière couche apprend (= configuration du prof)
%   E2  modeGel = "res5"  -> dernier bloc résiduel + dernière couche
%   E3  modeGel = "aucun" -> fine-tuning complet
%   E4  = E3 + augmentation de données

%% 0. Réglages de l'expérience (la seule section à modifier)
nomExperience = "E3_complet";  % nom libre, recopié dans le tableau des résultats
modeGel       = "aucun";           % "fc" | "res5" | "aucun" (voir ci-dessus)
augmentation  = false;          % true : flips, rotations, zooms aléatoires à chaque époque
nbEpoques     = 6;
learnRate     = 1e-4;           % 1e-3 pour "fc" ; 1e-4 pour "res5" et "aucun"
tailleBatch   = 32;
testRapide    = false;           % true : 500 images, 1 époque, rien n'est enregistré (vérification + chrono)
exporterTest  = true;          % true : écrit test_results.json (seulement pour le meilleur réseau)

%% 1. Données (chargées une seule fois, puis gardées en mémoire et dans le cache .mat)
imageSize    = [224 224];       % taille d'entrée de ResNet-18
trainPart    = 0.85;            % 85 % entraînement / 15 % validation
fichierCache = 'food11_224.mat';

if ~exist('XTrain', 'var')      % déjà en mémoire ? alors on ne recharge pas
    if isfile(fichierCache)
        fprintf('Chargement depuis %s...\n', fichierCache);
        load(fichierCache, 'XTrain', 'TTrain', 'XValid', 'TValid');
    else
        rng(0);                 % même découpage train/validation à chaque fois
        [XTrain, TTrain, XValid, TValid] = ...
            load_and_format_data(fullfile('food-11'), imageSize, trainPart);
        save(fichierCache, 'XTrain', 'TTrain', 'XValid', 'TValid', '-v7.3');
    end
    % Les images de test ne sont pas mises en cache : elles ne servent qu'à l'export JSON (section 5)
end
classNames = categories(TTrain);

% Données réellement utilisées pour l'entraînement (sous-ensemble en test rapide)
if testRapide
    rng(42);
    idx = randperm(size(XTrain, 4), 500);
    XTr = XTrain(:, :, :, idx);  TTr = TTrain(idx);
    nbEpoques = 1;
else
    XTr = XTrain;  TTr = TTrain;
end
fprintf('Entraînement sur %d images, validation sur %d images\n', size(XTr, 4), size(XValid, 4));

%% 2. Réseau : ResNet-18 pré-entraîné sur ImageNet, dernière couche remplacée (11 sorties)
% resnet18_11classes.mat : réseau exporté par un membre du groupe qui a l'add-on
% "Deep Learning Toolbox Model for ResNet-18 Network" (fichier .mat, non versionné).
% Sans ce fichier, on le charge directement depuis l'add-on.
if isfile("resnet18_11classes.mat")
    load("resnet18_11classes.mat", "net");
else
    net = imagePretrainedNetwork("resnet18", NumClasses=numel(classNames));
end

% Nom de la nouvelle dernière couche entièrement connectée
isFC   = arrayfun(@(L) isa(L, 'nnet.cnn.layer.FullyConnectedLayer'), net.Layers);
fcName = string(net.Layers(find(isFC, 1, 'last')).Name);

% Gel des couches : un "learn rate factor" de 0 empêche la mise à jour des poids.
% Les noms des couches du dernier bloc commencent par "res5" (convolutions) ou "bn5" (batch norm).
nbParamsEntrainables = 0;
for i = 1:height(net.Learnables)
    nomCouche = string(net.Learnables.Layer(i));
    nomParam  = string(net.Learnables.Parameter(i));
    switch modeGel
        case "fc",    entrainable = (nomCouche == fcName);
        case "res5",  entrainable = (nomCouche == fcName) || startsWith(nomCouche, ["res5", "bn5"]);
        case "aucun", entrainable = true;
        otherwise,    error('modeGel inconnu : %s', modeGel);
    end
    if ~entrainable
        net = setLearnRateFactor(net, nomCouche, nomParam, 0);
    else
        nbParamsEntrainables = nbParamsEntrainables + numel(net.Learnables.Value{i});
        % La nouvelle couche part de zéro : elle apprend 10x plus vite que les couches pré-entraînées
        if nomCouche == fcName && modeGel ~= "fc"
            net = setLearnRateFactor(net, nomCouche, nomParam, 10);
        end
    end
end
nbParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf('Paramètres : %d au total (%.2f M), dont %d entraînables (%.2f M)\n', ...
    nbParams, nbParams/1e6, nbParamsEntrainables, nbParamsEntrainables/1e6);

%% 3. Entraînement
iterParEpoque = ceil(size(XTr, 4) / tailleBatch);
options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    MaxEpochs=nbEpoques, ...
    MiniBatchSize=tailleBatch, ...
    Shuffle="every-epoch", ...
    LearnRateSchedule="piecewise", ...          % learning rate divisé par 10 pour la dernière époque
    LearnRateDropPeriod=max(1, nbEpoques - 1), ...
    LearnRateDropFactor=0.1, ...
    ValidationData={XValid, TValid}, ...
    ValidationFrequency=iterParEpoque, ...      % une validation par époque (coûteuse sur CPU)
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=true, VerboseFrequency=20);

tic;
if augmentation
    aug = imageDataAugmenter(RandXReflection=true, RandRotation=[-15 15], RandScale=[0.9 1.1]);
    donneesTrain = augmentedImageDatastore(imageSize, XTr, TTr, DataAugmentation=aug);
    trainedNet = trainnet(donneesTrain, net, "crossentropy", options);
else
    trainedNet = trainnet(XTr, TTr, net, "crossentropy", options);
end
dureeMin = toc / 60;

%% 4. Évaluation sur la validation : accuracy, F1 macro (score du serveur), frugalité
tic;
YValid = scores2label(minibatchpredict(trainedNet, XValid), classNames);
dureeValMin = toc / 60;

C = confusionmat(TValid, YValid);           % lignes = vraie classe, colonnes = classe prédite
precision = diag(C) ./ sum(C, 1)';
rappel    = diag(C) ./ sum(C, 2);
f1        = 2 * precision .* rappel ./ (precision + rappel);
f1(isnan(f1)) = 0;                          % classe jamais prédite -> F1 = 0

accuracy  = mean(YValid == TValid);
f1Macro   = mean(f1);
frugalite = f1Macro - 0.3 / (1 + 1e6 / nbParams);   % formule du sujet, paramètres en millions

disp(table(classNames, precision, rappel, f1, VariableNames=["classe", "precision", "rappel", "F1"]));
fprintf('%s : accuracy = %.4f | F1 macro = %.4f | frugalité estimée = %.4f | %.1f min\n', ...
    nomExperience, accuracy, f1Macro, frugalite, dureeMin);

fig = figure;
cm = confusionchart(TValid, YValid, ColumnSummary="column-normalized", RowSummary="row-normalized");
cm.Title = "ResNet-18 " + strrep(nomExperience, "_", " ") + " (validation)";

if testRapide
    % Estimation du temps d'une vraie époque (11 303 images), validation comprise
    tEpoque = (dureeMin - dureeValMin) * size(XTrain, 4) / size(XTr, 4) + dureeValMin;
    fprintf('\nTEST RAPIDE : 1 époque complète prendrait environ %.0f min sur cette machine.\n', tEpoque);
    fprintf('Rien n''a été enregistré. Mettre testRapide = false pour une vraie expérience.\n');
    return
end

%% 5. Enregistrements (seulement pour les vraies expériences)
exportgraphics(fig, "confusion_resnet18_" + nomExperience + ".png");

ligne = table(string(datetime("now")), nomExperience, modeGel, double(augmentation), nbEpoques, ...
    learnRate, tailleBatch, nbParams/1e6, nbParamsEntrainables/1e6, dureeMin, accuracy, f1Macro, frugalite, ...
    VariableNames=["date", "experience", "gel", "augmentation", "epoques", "lr", "batch", ...
                   "params_M", "params_entrainables_M", "duree_min", "accuracy", "F1_macro", "frugalite"]);
fichierRes = "resultats_resnet18.csv";
if isfile(fichierRes)
    writetable(ligne, fichierRes, WriteMode="append", WriteVariableNames=false);
else
    writetable(ligne, fichierRes);
end

% Réseau entraîné (fichier .mat : ignoré par git, à ne pas rendre)
save("resnet18_" + nomExperience + ".mat", "trainedNet", "classNames");

% Prédictions du test pour le serveur d'évaluation (meilleur réseau seulement)
% Les images de test sont lues directement depuis le disque, redimensionnées à la volée.
if exporterTest
    imdsTest  = imageDatastore(fullfile('food-11', 'test'));
    dsTest    = augmentedImageDatastore(imageSize, imdsTest, ColorPreprocessing="gray2rgb");
    [~, testNames] = cellfun(@fileparts, imdsTest.Files, 'UniformOutput', false);   % "0", "1", ...
    export_results_test(trainedNet, dsTest, testNames, classNames);   % minibatchpredict accepte un datastore
end