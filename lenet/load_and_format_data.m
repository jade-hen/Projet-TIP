function [XTrain, TTrain, XValid, TValid] = load_and_format_data(dataLocation, imageSize, trainPart)
    
    % on récupère les images
    images = imageDatastore(dataLocation, "IncludeSubfolders", true, 'LabelSource','foldernames');
    
    % on en prend certaines pour l'entraînement et d'autres pour la validation
    [trainingImages, validationImages] = splitEachLabel(images, trainPart, 'randomized');

    % on récupère les étiquettes
    TTrain = trainingImages.Labels;
    TValid = validationImages.Labels;
    
    % on redimensionne les images 
    trainingImagesResized = augmentedImageDatastore(imageSize, trainingImages);
    validationImagesResized = augmentedImageDatastore(imageSize, validationImages);
    
    % on récupère les images redimensionnées (readall renvoie une table dont la colonne '.input' contient les images) et on remet les data en dimension 4d ([Ligne du pixel, Colonne du pixel, Couleur R/G/B, Numéro de l'image])
    trainingData = readall(trainingImagesResized);
    validationData = readall(validationImagesResized);
    
    XTrain = cat(4, trainingData.input{:});
    XValid = cat(4, validationData.input{:});

end