function export_results_test(net, XTest, classNames)

    % on récupère les résultats (proba pour chaque classe)
    testResults = minibatchpredict(net, XTest);

    % on déduit les labels (prend celle de la proba la + élevée)
    labelsTestPred = scores2label(testResults, classNames);

    % choix du nom de fichier résultats
    jsonFileName = 'test_results.json';
    
    % on récupère les résultats
    testResults = string(labelsTestPred);

    % on crée la liste de numérotation
    num = string(0:(numel(testResults)-1))';

    % on associe chaque numéro avec résultat pour obtenir un json de la forme :
    % {
            %"0": "bread",
            %"1": "bread",
            %"2": "fried",
            %"3": "bread",
    % }
    resultsDict = dictionary(num, testResults);

    % on passe au format json avec PrettyPrint=true pour avoir les sauts de lignes et indentations comme le modèle
    jsonString = jsonencode(resultsDict, PrettyPrint=true);

    % on ouvre/crée le fichier en mode écriture
    fid = fopen(jsonFileName, 'w');

    % si le fichier a bien pu être créé/ouvert sans erreur
    if fid ~= -1
        fwrite(fid, jsonString, 'char');
        fclose(fid);

        % on affiche un log pour savoir que c'est fini
        fprintf('--> Fichier "%s" généré\n', jsonFileName);
    else
        % on affiche un log d'erreur
        warning('Echec de la génération du fichier JSON.');
    end


    

end