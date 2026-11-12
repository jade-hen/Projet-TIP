function confusion_matrix(net, XValid, TValid, nomFichier)
% Trace la matrice de confusion du réseau sur la validation et l'enregistre en PNG.
%   confusion_matrix(net, XValid, TValid)              -> matrice_de_confusion.png
%   confusion_matrix(net, XValid, TValid, nomFichier)  -> nomFichier (ex. "confusion_E1.png")

    if nargin < 4
        nomFichier = 'matrice_de_confusion.png';
    end

    % on récupère les résultats
    validResults = minibatchpredict(net, XValid);

    % on déduit les labels
    labelsValidPred = scores2label(validResults, categories(TValid));

    % on trace la matrice de confusion
    fig = figure;
    cm = confusionchart(TValid, labelsValidPred);
    cm.ColumnSummary = "column-normalized"; % Precision by predicted class.
    cm.RowSummary = "row-normalized";       % Recall by true class.
    cm.Title = "Validation Confusion Matrix";

    exportgraphics(fig, nomFichier)

    fprintf('Matrice de confusion enregistrée dans %s\n', nomFichier);

end