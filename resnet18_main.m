%% 0. Réglages
nomExperience   = "E6_coupe_res4";
modeGel         = "aucun";     % "fc" (dernière couche seule), "res5" (res5 + dernière couche), "aucun" (tout le réseau)
coupure         = "res4";      % "aucune", "res4" (retire res5), "res3" (retire res4 et res5)
augmentation    = false;       % symétrie horizontale, rotation de ±15°, zoom de 0,9 à 1,1
regulL2         = 1e-3;
nbEpoques       = 6;
learnRate       = 1e-4;
tailleBatch     = 32;
testRapide      = false;       % si true alors 500 images et 1 époque, pour vérifier que le script fonctionne
exporterTest    = true;        % si true alors écrit test_results_<nomExperience>.json
afficherCourbes = true;        % fenêtre Training Progress de MATLAB

%% 1. Données
imageSize    = [224 224];      % taille d'entrée de ResNet-18
trainPart    = 0.85;           % 85 % entraînement, 15 % validation
fichierCache = 'food11_224.mat';

if ~exist('XTrain', 'var')
    if isfile(fichierCache)
        load(fichierCache, 'XTrain', 'TTrain', 'XValid', 'TValid');
    else
        rng(0);                % même découpage à chaque fois
        [XTrain, TTrain, XValid, TValid] = load_and_format_data('food-11', imageSize, trainPart);
        save(fichierCache, 'XTrain', 'TTrain', 'XValid', 'TValid', '-v7.3');
    end
end
classNames = categories(TTrain);

if testRapide
    rng(1);
    idx = randperm(size(XTrain, 4), 500);
    XTr = XTrain(:, :, :, idx);
    TTr = TTrain(idx);
    nbEpoques = 1;
else
    XTr = XTrain;
    TTr = TTrain;
end
fprintf('Entraînement : %d images, validation : %d images\n', size(XTr, 4), size(XValid, 4));

%% 2. Réseau
% ResNet-18 avec une dernière couche à 11 sorties
if isfile("resnet18_11classes.mat")
    load("resnet18_11classes.mat", "net");
else
    net = imagePretrainedNetwork("resnet18", NumClasses=numel(classNames));
end

% Réseau coupé (on supprime les étages après la coupure et on ajoute une nouvelle sortie)
if coupure ~= "aucune"
    derniereCouche = coupure + "b_relu";
    noms = string({net.Layers.Name});
    iCoupe = find(noms == derniereCouche);
    net = removeLayers(net, noms(iCoupe+1:end));
    net = addLayers(net, [
        globalAveragePooling2dLayer(Name="gap_food")
        fullyConnectedLayer(numel(classNames), Name="fc_food")
        softmaxLayer(Name="prob_food")]);
    net = connectLayers(net, derniereCouche, "gap_food");
    net = initialize(net);
end

% Gel des couches
fcName = string(net.Learnables.Layer(end));
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
    if entrainable
        nbParamsEntrainables = nbParamsEntrainables + numel(net.Learnables.Value{i});
        % la nouvelle couche part de zéro, elle apprend 10 fois plus vite
        if nomCouche == fcName && modeGel ~= "fc"
            net = setLearnRateFactor(net, nomCouche, nomParam, 10);
        end
    else
        net = setLearnRateFactor(net, nomCouche, nomParam, 0);
    end
end
nbParams = sum(cellfun(@numel, net.Learnables.Value));
fprintf('Paramètres : %d (%.2f M), dont %d entraînables (%.2f M)\n', ...
    nbParams, nbParams/1e6, nbParamsEntrainables, nbParamsEntrainables/1e6);

%% 3. Entraînement
iterParEpoque = ceil(size(XTr, 4) / tailleBatch);
if afficherCourbes
    affichage = "training-progress";
else
    affichage = "none";
end

options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    MaxEpochs=nbEpoques, ...
    MiniBatchSize=tailleBatch, ...
    Shuffle="every-epoch", ...
    LearnRateSchedule="piecewise", ...      % learning rate divisé par 10 à la dernière époque
    LearnRateDropPeriod=max(1, nbEpoques - 1), ...
    LearnRateDropFactor=0.1, ...
    L2Regularization=regulL2, ...
    ValidationData={XValid, TValid}, ...
    ValidationFrequency=iterParEpoque, ...  % une validation par époque
    Metrics="accuracy", ...
    Plots=affichage, ...
    Verbose=true, VerboseFrequency=20);

tic;
if augmentation
    % Augmentation des données
    aug = imageDataAugmenter(RandXReflection=true, RandRotation=[-15 15], RandScale=[0.9 1.1]);
    dsTrain = augmentedImageDatastore(imageSize, XTr, TTr, DataAugmentation=aug);
    [trainedNet, info] = trainnet(dsTrain, net, "crossentropy", options);
else
    [trainedNet, info] = trainnet(XTr, TTr, net, "crossentropy", options);
end
dureeMin = toc / 60;

%% 4. Évaluation sur la validation
YValid = scores2label(minibatchpredict(trainedNet, XValid), classNames);

C = confusionmat(TValid, YValid);
precision = diag(C) ./ sum(C, 1)';
rappel    = diag(C) ./ sum(C, 2);
f1        = 2 * precision .* rappel ./ (precision + rappel);
f1(isnan(f1)) = 0;

accuracy = mean(YValid == TValid);
f1Macro  = mean(f1);

% F1 calculé comme le serveur : précision et rappel moyens pondérés par l'inverse de la fréquence de chaque classe, puis moyenne harmonique ; frugalité = formule du sujet
poids     = (1 ./ sum(C, 2)) / sum(1 ./ sum(C, 2));
precPond  = sum(poids .* precision, 'omitnan');
rappPond  = sum(poids .* rappel);
f1Serveur = 2 * precPond * rappPond / (precPond + rappPond);
frugalite = f1Serveur - 0.3 / (1 + 1e6 / nbParams);

fprintf('%s : accuracy = %.4f | F1 macro = %.4f | F1 serveur = %.4f | frugalité = %.4f | %.1f min\n', ...
    nomExperience, accuracy, f1Macro, f1Serveur, frugalite, dureeMin);

fig = figure;
cm = confusionchart(TValid, YValid, ColumnSummary="column-normalized", RowSummary="row-normalized");
cm.Title = "ResNet-18 " + strrep(nomExperience, "_", " ") + " (validation)";

if testRapide
    return                      % si test rapide alors aucun fichier enregistré
end

%% 5. Enregistrement des résultats
exportgraphics(fig, "confusion_resnet18_" + nomExperience + ".png");

ligne = table(string(datetime("now")), nomExperience, modeGel, double(augmentation), nbEpoques, ...
    learnRate, tailleBatch, nbParams/1e6, nbParamsEntrainables/1e6, dureeMin, accuracy, f1Macro, ...
    frugalite, f1Serveur, regulL2, ...
    VariableNames=["date", "experience", "gel", "augmentation", "epoques", "lr", "batch", ...
                   "params_M", "params_entrainables_M", "duree_min", "accuracy", "F1_macro", ...
                   "frugalite", "F1_serveur", "L2"]);
fichierRes = "resultats_resnet18.csv";
if isfile(fichierRes)
    writetable(ligne, fichierRes, WriteMode="append", WriteVariableNames=false);
else
    writetable(ligne, fichierRes);
end

save("resnet18_" + nomExperience + ".mat", "trainedNet", "classNames", "info");
writetable(info.TrainingHistory,   "historique_train_" + nomExperience + ".csv");
writetable(info.ValidationHistory, "historique_valid_" + nomExperience + ".csv");

% Prédictions sur le jeu de test pour le serveur d'évaluation
if exporterTest
    imdsTest = imageDatastore(fullfile('food-11', 'test'));
    dsTest   = augmentedImageDatastore(imageSize, imdsTest, ColorPreprocessing="gray2rgb");
    [~, testNames] = cellfun(@fileparts, imdsTest.Files, 'UniformOutput', false);
    export_results_test(trainedNet, dsTest, testNames, classNames);
    movefile('test_results.json', "test_results_" + nomExperience + ".json");
    fprintf('Prédictions du test : test_results_%s.json (%.6f M paramètres à déclarer)\n', ...
        nomExperience, nbParams/1e6);
end