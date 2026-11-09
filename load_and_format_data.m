function [XTrain, TTrain, XValid, TValid, XTest] = load_and_format_data(dataLocation, imageSize, trainPart)
    
    % on récupère les images d'entraînement
    trainPath = fullfile(dataLocation, 'train');
    images = imageDatastore(trainPath, "IncludeSubfolders", true, 'LabelSource','foldernames');
    
    % on en prend certaines pour l'entraînement et d'autres pour la validation
    [trainingImages, validationImages] = splitEachLabel(images, trainPart, 'randomized');

    % on récupère les étiquettes
    TTrain = trainingImages.Labels;
    TValid = validationImages.Labels;
    
    % on charge les images redimensionnées dans des tableaux 4D
    XTrain = readImagesToArray(trainingImages, imageSize);
    XValid = readImagesToArray(validationImages, imageSize);
    
    % on récupère les images de test
    testPath = fullfile(dataLocation, 'test');
    testImages = imageDatastore(testPath, "IncludeSubfolders", true);
    
    XTest = readImagesToArray(testImages, imageSize);

end

function X = readImagesToArray(imds, imageSize)
    n = numel(imds.Files);
    X = zeros([imageSize 3 n], "uint8");
    augds = augmentedImageDatastore(imageSize, imds, ColorPreprocessing="gray2rgb");
    augds.MiniBatchSize = 64;
    k = 0;
    while hasdata(augds)
        batch = read(augds);
        for i = 1:numel(batch.input)
            k = k + 1;
            X(:,:,:,k) = batch.input{i};
        end
    end
end
