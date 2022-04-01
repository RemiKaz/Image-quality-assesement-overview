# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 18:13:17 2022

@author: remik
"""

from PIL import Image,ImageOps
import numpy as np
from MR_perceptual import test_network,mrpl
import torch 
import skimage
import matplotlib.pyplot as plt
import torch.nn.functional as F

### Partie 1 : charger / afficher l'image

# Chargement

image = Image.open('burger.jpg') # Chargement de l'image
image_gray = ImageOps.grayscale(image) # Conversion en nuance de gris

image_gray_numpy = np.array(image_gray) # Conversion en array numpy (image grise)
image_numpy = np.array(image) # Conversion en array numpy (image couleur)


# Affichage

# Image en nuance de gris

'''print("Image nuances de gris \n", image_gray_numpy) # Affichage du tuple
print("Taille de l\'image' \n", image_gray_numpy.shape) # Affichage de la taille du tuple
image_gray.show() # Affichage de l'image

# Image couleur

print('Image nuances de gris \n', image_numpy) # Affichage du tuple
print('Image nuances de gris \n', image_numpy.shape) # Affichage de la taille du tuple
image.show() # Affichage de l'image'''

### Partie 2 : Bruitage de l'image

# Paramètres du bruit gaussien

mean = 0 # Moyenne
sigma = 50 # Ecart type

# Ajout du bruit

# Fonction d'ajout

def add_gaussian_noise(img,mean,sigma):

    gaussian_noise = np.random.normal(mean, sigma,img.shape) # Creation du bruit
    image_noise = np.clip(img + gaussian_noise,0,255).astype(np.uint8) # Ajout du bruit

    return image_noise

# Creation des images bruitées
    
image_noise = add_gaussian_noise(image_numpy,mean,sigma)

# Affichage des images bruitées

'''image_bruit = Image.fromarray(image_noise) # Conversion en image PIL 
image_bruit.show() # Affichage'''

### Partie 3 : SSIM/PSNR

## Bruit gaussien

# SSIM : (Note : multichannel=True est juste un paramètre pour indiquer que l'image est sur plusieurs chanels, plus d'infos sur
# https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity)
ssim = skimage.metrics.structural_similarity(image_noise,image_numpy,multichannel=True)

# SSIM : (plus d'infos sur https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio)
psnr = skimage.metrics.peak_signal_noise_ratio(image_noise,image_numpy)

'''print("Métriques pour le bruit gaussien \n","SSIM : ",ssim,"\n PSNR : ",psnr)
'''
## Images BAPS dataset (https://github.com/richzhang/PerceptualSimilarity)

#Importation des images

image_ref = Image.open('ex_ref.png')
image_1 = Image.open('ex_p0.png')
image_2 = Image.open('ex_p1.png')

# Conversion en array

image_ref_numpy = np.array(image_ref)
image_1_numpy = np.array(image_1)
image_2_numpy = np.array(image_2)

# SSIM

ssim1 = skimage.metrics.structural_similarity(image_ref_numpy,image_1_numpy,multichannel=True)
ssim2 = skimage.metrics.structural_similarity(image_ref_numpy,image_2_numpy,multichannel=True)

# PSNR

psnr1 = skimage.metrics.peak_signal_noise_ratio(image_ref_numpy,image_1_numpy)
psnr2 = skimage.metrics.peak_signal_noise_ratio(image_ref_numpy,image_2_numpy)

'''print("Métriques pour les images du BAPS dataset \n","SSIM image 1 : ",ssim1,"\n SSIM image 2 : ",ssim2,"\n PSNR image 1 : ",psnr1,"\n PSNR image 2 : ",psnr2)
'''
### Partie 4 : Pouvoir perceptuel des réseaux de neurones

## Chargement des images en tant que tenseurs

tensor_ref = mrpl.im2tensor(mrpl.load_image('ex_ref.png'))
tensor_im1 = mrpl.im2tensor(mrpl.load_image('ex_p0.png'))
tensor_im2 = mrpl.im2tensor(mrpl.load_image('ex_p1.png'))

tensor_ref_up = F.interpolate(tensor_ref, size=(256, 256), mode='bicubic', align_corners=False)
tensor_im1_up = F.interpolate(tensor_im1, size=(256, 256), mode='bicubic', align_corners=False)
tensor_im2_up = F.interpolate(tensor_im2, size=(256, 256), mode='bicubic', align_corners=False)

# Création de la fonction d'objet MRPL, qui contient un réseaux de neurones (alexnet)

loss_fn = mrpl.MRPL(net='alex', spatial=False,mrpl=True,verbose=False)  

'''# plot_features_maps > Fonction qui affiche des couches (features maps) d'un bloc. 
# In   features : features extraites d'un bloc du réseau 
#      n_channels : nombre de couches que l'on veut afficher
#      n_bloc : numero du bloc à extraire
#      title : titre du tracé'''

def plot_features_maps(features,n_chanels,n_bloc,title):
    plt.suptitle(title)
    F = features[n_bloc] # On prends par exemple la sortie du premier bloc. Plus le bloc est profond, plus les features sont de haut niveau
    for i in range (0,n_chanels): 
        im = np.array(F[0,i,:,:])
        im = im.astype(np.uint8)
        plt.subplot(5,5,i+1)
        plt.axis('off')
        plt.imshow(im)
        
'''# plot_features_maps > Fonction qui affiche des couches (features maps) d'un bloc. 
# In   features : features extraites d'un bloc du réseau 
#      n_channels : nombre de couches que l'on veut afficher
#      l_blocs : numero des blocs pris en compte
# Out  D : distance calculée'''
        
def distance_features_maps(features1,features2,l_blocs):
    D = 0 
    #for i in range (0,len(features1)-8):
    for i in l_blocs:
        
        features1_arr = np.array(features1[i]).astype(np.uint8)
        features2_arr = np.array(features2[i]).astype(np.uint8)
        D+=np.linalg.norm(features1_arr-features2_arr)
        
    return D
## Extraction des features de l'image de reference

features_ref_up = loss_fn.net.forward(tensor_ref_up) 

plt.figure(1)
n_chanels = 20 # On affiche (arbitrairement) 20 couches (features maps) de ce bloc
n_bloc = 0 

#plot_features_maps(features_ref_up,n_chanels,n_bloc,"Image de référence") # Tracé
        
'''## Extraction des features de l'image 1      
        
features_img1 = loss_fn.net.forward(tensor_im1) # Appel du réseau préentrainé retourne une liste de tenseurs qui correspond a chaque sortie de block
features_img1_up = loss_fn.net.forward(tensor_im1_up)

plt.figure(2)
n_chanels = 20 # On affiche (arbitrairement) 20 couches (features maps) de ce bloc

plot_features_maps(features_img1_up,n_chanels,n_bloc,"Image 1") # Tracé

## Extraction des features de l'image 2
        
features_img2_up = loss_fn.net.forward(tensor_im2_up) 

plt.figure(3)
n_chanels = 20 # On affiche (arbitrairement) 20 couches (features maps) de ce bloc

plot_features_maps(features_img2_up,n_chanels,n_bloc,"Image 2") # Tracé'''

## Distacnce naïve

features_img1 = loss_fn.net.forward(tensor_im1) # Appel du réseau préentrainé retourne une liste de tenseurs qui correspond a chaque sortie de block
features_img2 = loss_fn.net.forward(tensor_im2) # Appel du réseau préentrainé retourne une liste de tenseurs qui correspond a chaque sortie de block
features_ref = loss_fn.net.forward(tensor_ref) # Appel du réseau préentrainé retourne une liste de tenseurs qui correspond a chaque sortie de block

D1_direct = distance_features_maps(features_ref,features_img1,[4,5])
D2_direct = distance_features_maps(features_ref,features_img2,[4,5])

print("Distance features maps direct \n","Image 1 : ",D1_direct,"\n Image 2 : ",D2_direct)

### Partie 5 : Calcul de la métrique mrpl

D1_MRPL = test_network.MRPL_2_images('ex_p0.png','ex_ref.png')
D2_MRPL = test_network.MRPL_2_images('ex_p1.png','ex_ref.png')

print("MRPL Distance \n","Image 1 : ",D1_MRPL,"\n Image 2 : ",D2_MRPL)






