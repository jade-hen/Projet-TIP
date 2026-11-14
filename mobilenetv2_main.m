clear; clc; close all;

%% 0. Paramètres de l'expérience 
nomExp       = "E5";   % nom de l'expérience
dossierSortie = "resultats_E5";    % dossier résultats
mode         = "ft";    % "ft" = fine-tuning complet | "gel" = couches figées 
augmentation = false;   % true = augmentation de données 
nbEpochs     = 5;
l2       = 1e-4;    % régularisation L2 (1e-4 par défaut ; 1e-3 pour plus forte)
patience = Inf;     % arrêt anticipé : ex. 3 = stop si la validation ne s'améliore pas 3 fois de suite | Inf = l'entraînement va jusqu'au bout

if ~isfolder(dossierSortie), mkdir(dossierSortie); end

%% 1. Chargement du modele pré-entrainé
numClasses = 11;
modelName  = "mobilenetv2";
net = imagePretrainedNetwork(modelName, NumClasses=numClasses);

% taille d'entrée (224x224 pour mobilenetv2)
inputSize = net.Layers(1).InputSize;
imageSize = inputSize(1:2);

%% 2. Chargement et formatage des données
dataLocation = "food-11";   % contient les sous-dossiers "train" et "test"
trainPart    = 0.85;         % 85 % entrainement, 15 % validation
rng(0);                     % même découpage à chaque fois

cacheFile = "food11_" + imageSize(1) + "_" + round(trainPart*100) + ".mat";
if isfile(cacheFile)
    load(cacheFile, "XTrain", "TTrain", "XValid", "TValid", "XTest", "testNames");
else
    [XTrain, TTrain, XValid, TValid, XTest, testNames] = ...
        load_and_format_data(dataLocation, imageSize, trainPart);
    save(cacheFile, "XTrain", "TTrain", "XValid", "TValid", "XTest", "testNames", "-v7.3");
end
classNames = categories(TTrain);

%% 3. Gel des couches selon "mode"
L = net.Learnables;
derniere = L.Layer(end);                 % nouvelle couche de classification : jamais gelée

if mode == "gel"
    for i = 1:height(L)
        if L.Layer(i) ~= derniere
            net = setLearnRateFactor(net, L.Layer(i) + "/" + L.Parameter(i), 0);
        end
    end
    lr = 1e-3;   % seule la fin du réseau apprend
else
    lr = 1e-4;   % fine-tuning
end

% nombre total de paramètres du réseau
nbParams       = sum(cellfun(@numel, L.Value));
estEntrainable = (L.Layer == derniere) | (mode == "ft");
nbParamsEntr   = sum(cellfun(@numel, L.Value(estEntrainable)));
fprintf("Mode : %s | augmentation : %d | paramètres entraînables : %.3f M sur %.2f M\n", ...
    mode, augmentation, nbParamsEntr/1e6, nbParams/1e6);

%% 4. Options d'entraînement
miniBatchSize = 16;
itersPerEpoch = floor(size(XTrain,4) / miniBatchSize);

options = trainingOptions("adam", ...
    InitialLearnRate=lr, ...
    MaxEpochs=nbEpochs, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    L2Regularization=l2, ...
    ValidationPatience=patience, ...
    ValidationData={XValid, TValid}, ...     
    ValidationFrequency=itersPerEpoch, ...
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=true, VerboseFrequency=50);

%% 5. Entraînement (avec ou sans augmentation de données)
tic;
if augmentation
    augmenter = imageDataAugmenter( ...
        RandXReflection=true, ...              % miroir horizontal
        RandRotation=[-15 15], ...             % rotation aléatoire (degrés)
        RandScale=[0.9 1.1], ...               % zoom aléatoire
        RandXTranslation=[-20 20], ...         % décalage horizontal (pixels)
        RandYTranslation=[-20 20]);            % décalage vertical (pixels)
    dsTrain = augmentedImageDatastore(imageSize, XTrain, TTrain, DataAugmentation=augmenter);
    [trainedNet, info] = trainnet(dsTrain, net, "crossentropy", options);
else
    [trainedNet, info] = trainnet(XTrain, TTrain, net, "crossentropy", options);
end
tTrain = toc;

%% 6. Évaluation sur la validation : accuracy, F1, frugalité
tic;
YValid = scores2label(minibatchpredict(trainedNet, XValid), classNames);
tInference = toc;
msParImage = 1000 * tInference / numel(TValid);

C = confusionmat(TValid, YValid);           % lignes = vraie classe, colonnes = classe prédite
precision = diag(C) ./ sum(C, 1)';
rappel    = diag(C) ./ sum(C, 2);
f1        = 2 * precision .* rappel ./ (precision + rappel);
f1(isnan(f1)) = 0;                          % classe jamais prédite -> F1 = 0

accuracy  = mean(YValid == TValid);
f1Macro   = mean(f1);                       % moyenne simple des F1 par classe (F1 macro standard)

% Variante "serveur" : précision et rappel moyens pondérés par l'inverse de la
% fréquence de chaque classe (poids normalisés), puis moyenne harmonique.

poids     = (1 ./ sum(C, 2)) / sum(1 ./ sum(C, 2));
precPond  = sum(poids .* precision, 'omitnan');
rappPond  = sum(poids .* rappel);
f1Serveur = 2 * precPond * rappPond / (precPond + rappPond);

% Score de frugalité du sujet : Frug = P - w / (1 + 1/R)
%   P = F1, w = 0.3, R = nombre de paramètres EN MILLIONS

w = 0.3;
R = nbParams / 1e6;
frugalite      = f1Serveur - w / (1 + 1/R);
frugaliteMacro = f1Macro   - w / (1 + 1/R);

disp(table(classNames, precision, rappel, f1, VariableNames=["classe", "precision", "rappel", "F1"]));
fprintf('%s : accuracy = %.4f | F1 macro = %.4f | F1 (calcul du serveur) = %.4f\n', ...
    nomExp, accuracy, f1Macro, f1Serveur);
fprintf('Frugalité (F1 serveur) = %.4f | Frugalité (F1 macro) = %.4f | R = %.3f M params\n', ...
    frugalite, frugaliteMacro, R);

%% 8. Journal des essais (une ligne par exécution, pour comparer ft / gel / augmentation)
ligne = table(string(datetime("now")), string(nomExp), mode, augmentation, nbEpochs, accuracy, f1Macro, ...
    f1Serveur, frugalite, tTrain/60, nbParamsEntr/1e6, msParImage,  ...
    'VariableNames', ["date" "nom_exp" "mode" "augmentation" "epochs" "accuracy" "f1_macro" ...
                      "f1_serveur" "frugalite" "temps_min" "params_entr_M" "ms_par_image"]);
logFile = "resultats_essais.csv";
writetable(ligne, logFile, WriteMode="append", WriteVariableNames=~isfile(logFile));
fprintf("\nRésultat ajouté à %s\n", logFile);

%% 8b. Export des graphiques : courbes d'apprentissage et matrice de confusion
% --- Courbes (perte et accuracy, entraînement + validation) ---
TH = info.TrainingHistory;
VH = info.ValidationHistory;

figC = figure("Visible", "off", "Position", [100 100 1100 450]);
tl = tiledlayout(figC, 1, 2, "TileSpacing", "compact");
title(tl, "Courbes d'apprentissage - " + nomExp, "Interpreter", "none");

nexttile(tl);
plot(TH.Iteration, TH.Loss, "DisplayName", "Entraînement"); hold on;
if ~isempty(VH)
    ok = ~isnan(VH.Loss);
    plot(VH.Iteration(ok), VH.Loss(ok), "o-", "LineWidth", 1.5, "DisplayName", "Validation");
end
xlabel("Itération"); ylabel("Perte (crossentropy)"); grid on; legend("Location", "best");

nexttile(tl);
colAcc = intersect(["Accuracy" "accuracy"], string(TH.Properties.VariableNames));
plot(TH.Iteration, TH.(colAcc(1)), "DisplayName", "Entraînement"); hold on;
if ~isempty(VH)
    ok = ~isnan(VH.(colAcc(1)));
    plot(VH.Iteration(ok), VH.(colAcc(1))(ok), "o-", "LineWidth", 1.5, "DisplayName", "Validation");
end
xlabel("Itération"); ylabel("Accuracy (%)"); grid on; legend("Location", "best");

fichierCourbe = fullfile(dossierSortie, nomExp + "_courbe.png");
exportgraphics(figC, fichierCourbe, "Resolution", 200);
close(figC);

% --- Matrice de confusion (validation) ---
figM = figure("Visible", "off", "Position", [100 100 800 700]);
cc = confusionchart(figM, TValid, YValid);
cc.Title = "Matrice de confusion (validation) - " + nomExp;
cc.RowSummary = "row-normalized";
cc.ColumnSummary = "column-normalized";
fichierConf = fullfile(dossierSortie, nomExp + "_confusion.png");
exportgraphics(figM, fichierConf, "Resolution", 200);
close(figM);

fprintf("Graphiques enregistrés : %s et %s\n", fichierCourbe, fichierConf);

%% 9. Predictions sur le jeu de test + export JSON
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

%% 10. Sauvegarde
save("food11_trainedNet_" + nomExp + ".mat", "trainedNet", "classNames");