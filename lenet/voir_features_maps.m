% voir_features_maps.m : feature maps de la première couche du réseau final (essai E6)
% Reprend Visualize_features.m du TP. Les lignes modifiées sont signalées par des flèches ←
% Il faut lancer main_lenet.m avant, car ce script utilise net, imgDataVal et labelsVal
%
% predict(..., Outputs=layer) renvoie ce que produit la photo choisie dans cette couche :
% chaque vignette en niveaux de gris correspond à la sortie d'un des filtres appris

layerName = "relu1";                                    % ← modifié : ReLU après conv1 et batchnorm1 (creer_cnn_gap.m)
imageIndex = 1;                                         % numéro de la photo dans la validation
% imageIndex = find(labelsVal == "soup", 1);            % ← ajouté : variante pour choisir une classe
image = imgDataVal(:, :, :, imageIndex);                % ← modifié : test → validation

% La couche choisie est la sortie ReLU de la première convolution
% predict attend des nombres réels, d'où le single()
featureMaps = predict(net, single(image), Outputs=layerName);
if isa(featureMaps, "dlarray")
    featureMaps = extractdata(featureMaps);
end
featureMaps = squeeze(featureMaps(:, :, :, 1));

figure
layout = tiledlayout(3, 3, TileSpacing="compact", Padding="compact");
title(layout, "Food-11 : image d'entrée et feature maps de la première couche");   % ← modifié : Food-11

nexttile
imshow(image)
title("Image d'entrée : " + string(labelsVal(imageIndex)))   % ← modifié : affiche la vraie classe

for channel = 1:8                                       % 16 filtres au total : mettre 9:16 pour voir les autres
    nexttile
    imagesc(featureMaps(:, :, channel))
    axis image off
    colormap(gca, gray(256))
    title("Filtre " + channel)                          % ← modifié : « Channel » traduit
end
