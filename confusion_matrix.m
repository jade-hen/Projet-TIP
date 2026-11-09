function confusion_matrix(net, XValid, TValid)

    % on récupère les résultats
    validResults = minibatchpredict(net, XValid);

    % on déduit les labels
    labelsValidPred = scores2label(validResults, categories(TValid));
    
    % on trace la matrice de confusion
    cm = confusionchart(TValid, labelsValidPred);
    cm.ColumnSummary = "column-normalized"; % Precision by predicted class.
    cm.RowSummary = "row-normalized";       % Recall by true class.
    cm.Title = "Validation Confusion Matrix";

end