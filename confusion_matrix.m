function confusion_matrix(net, XValid, TValid)

    % on récupère les résultats
    scoresValid = minibatchpredict(net, XValid);

    % on déduit les labels
    labelsPred = scores2label(scoresValid, categories(TValid));
    
    % on trace la matrice de confusion
    cm = confusionchart(TValid, labelsPred);
    cm.ColumnSummary = "column-normalized"; % Precision by predicted class.
    cm.RowSummary = "row-normalized";       % Recall by true class.
    cm.Title = "Validation Confusion Matrix";

end