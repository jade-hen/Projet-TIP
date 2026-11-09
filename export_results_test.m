function export_results_test(net, XTest, testNames, classNames)

    % on récupère les résultats (proba pour chaque classe)
    testResults = minibatchpredict(net, XTest);

    % on déduit les labels (prend celle de la proba la + élevée)
    labelsTestPred = scores2label(testResults, classNames);

    % choix du nom de fichier résultats
    jsonFileName = 'test_results.json';
    
    % on récupère les résultats
    testResults = string(labelsTestPred);


    % on associe chaque numéro (nom de l'image) avec résultat pour obtenir un json de la forme :
    % {
            %"0": "bread",
            %"1": "bread",
            %"2": "fried",
            %"3": "bread",
    % }
    % on trie pour avoir les résultats dans l'ordre en mettant x devant car
    % sinon trie 0 1 10 100 etc et struct ne peut être un chiffre c'est
    % donc une astuce
    imageNumber = cellfun(@str2double, testNames);
    [~, sortIdx] = sort(imageNumber);
    sortedNames = testNames(sortIdx);
    sortedResults = testResults(sortIdx);
    results = cell2struct(cellstr(sortedResults), strcat('x', sortedNames), 1);
    
    % on convertit en json et on retire le x de l'astuce
    jsonString = jsonencode(results, PrettyPrint=true);
    jsonString = regexprep(jsonString, '"x(\d+)":', '"$1":');

    % on ouvre/crée le fichier en mode écriture
    fid = fopen(jsonFileName, 'w');

    % si le fichier a bien pu être créé/ouvert sans erreur
    if fid ~= -1
        fwrite(fid, jsonString, 'char');
        fclose(fid);

        % on affiche un log pour savoir que c'est fini
        fprintf('Fichier "%s" généré\n', jsonFileName);
    else
        % on affiche un log d'erreur
        warning('Echec de la génération du fichier JSON.');
    end


    

end