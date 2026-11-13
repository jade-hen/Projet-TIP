% trace_courbes.m : courbes d'entraînement des expériences ResNet-18 pour le rapport
%
% Lit les fichiers historique_train_<nom>.csv et historique_valid_<nom>.csv écrits par
% resnet18_main.m, et trace pour chaque expérience :
%   - la perte et l'accuracy d'entraînement (lissées sur 20 itérations),
%   - la perte et l'accuracy de validation (un point par époque).
% Un écart qui se creuse entre entraînement et validation = surapprentissage.

experiences = ["E1_fc_seule", "E3_complet"];   % <-- noms des expériences à comparer
lissage = 20;                                    % fenêtre de lissage, en itérations

fig = figure(Position=[100 100 1100 420]);
tl = tiledlayout(fig, 1, 2, TileSpacing="compact");
axPerte = nexttile(tl); hold(axPerte, "on"); grid(axPerte, "on");
axAcc   = nexttile(tl); hold(axAcc, "on");   grid(axAcc, "on");
couleurs = lines(numel(experiences));

for k = 1:numel(experiences)
    nom = experiences(k);
    fTrain = "historique_train_" + nom + ".csv";
    fValid = "historique_valid_" + nom + ".csv";
    if ~isfile(fTrain) || ~isfile(fValid)
        warning("Historique introuvable pour %s : expérience ignorée", nom);
        continue
    end
    H = readtable(fTrain);
    V = readtable(fValid);

    % Abscisse en époques (1 époque = nombre d'itérations de la première époque)
    iterParEpoque = max(H.Iteration(H.Epoch == 1));
    xT = H.Iteration / iterParEpoque;
    xV = V.Iteration / iterParEpoque;
    c = couleurs(k, :);
    etiquette = strrep(nom, "_", " ");

    plot(axPerte, xT, movmean(H.TrainingLoss, lissage), Color=c, LineWidth=1.2, ...
        DisplayName=etiquette + " (entraînement)");
    plot(axPerte, xV, V.ValidationLoss, "o--", Color=c, LineWidth=1.5, MarkerFaceColor=c, ...
        DisplayName=etiquette + " (validation)");

    plot(axAcc, xT, movmean(H.TrainingAccuracy, lissage), Color=c, LineWidth=1.2, ...
        DisplayName=etiquette + " (entraînement)");
    plot(axAcc, xV, V.ValidationAccuracy, "o--", Color=c, LineWidth=1.5, MarkerFaceColor=c, ...
        DisplayName=etiquette + " (validation)");
end

xlabel(axPerte, "Époque");  ylabel(axPerte, "Perte (entropie croisée)");  title(axPerte, "Perte");
xlabel(axAcc, "Époque");    ylabel(axAcc, "Accuracy (%)");                title(axAcc, "Accuracy");
ylim(axAcc, [0 100]);
legend(axAcc, Location="southeast");

exportgraphics(fig, "courbes_resnet18.png", Resolution=200);
fprintf("Courbes enregistrées dans courbes_resnet18.png\n");