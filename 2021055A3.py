import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as calculate_ssim
import torchvision
from torchvision import transforms
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from __init__ import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
device = DEVICE

####----------------------------------------AlteredMNIST Start----------------------------------------####


def weighted_average_fusion(image1, image2, weight1=0.5, weight2=0.5):
    """
    Fuse two images with specified weights.
    
    Parameters:
    - image1, image2: The images to fuse.
    - weight1, weight2: The weights for each image.
    
    Returns:
    - Fused image.
    """
    # use numpy to perform element-wise weighted average
    fused_image = weight1 * image1 + weight2 * image2
    return fused_image

def fusion_with_noise(image1, image2, weight1=0.5, weight2=0.5, noise_level=0.5):
    """
    Fuse two images with specified weights and add Gaussian noise.
    
    Parameters:
    - image1, image2: The images to fuse.
    - weight1, weight2: The weights for each image.
    - noise_level: Standard deviation of the Gaussian noise to add.
    
    Returns:
    - Noisy fused image.
    """
    fused_image = weighted_average_fusion(image1, image2, weight1, weight2)
    noise = np.random.normal(0, noise_level, fused_image.shape)
    noisy_fused_image = np.clip(fused_image + noise, 0, 255)
    # noisy_fused_image = fused_image + noise

    return noisy_fused_image

def ensure_4d(tensor):
    """
    Ensure the tensor is 4-dimensional by adding missing batch and/or channel dimensions.
    """
    if tensor.dim() == 2:  # shape [H, W]
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Convert to [1, 1, H, W]
    elif tensor.dim() == 3:  # shape [C, H, W]
        tensor = tensor.unsqueeze(0)  # Convert to [1, C, H, W]
    return tensor

def gaussian_pyramid(image, levels):
    """
    Generate a Gaussian pyramid for an image.
    Ensures the image is 4D [batch_size, channels, height, width] before downsampling.
    """
    # Ensure the input image is 4D
    if image.dim() == 2:  # If image is [H, W], reshape to [1, 1, H, W]
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:  # If image is [C, H, W], reshape to [1, C, H, W]
        image = image.unsqueeze(0)
    
    pyramids = [image]  # The first level of the pyramid is the original image
    for i in range(1, levels):
        # Apply average pooling to downsample the image
        downsampled = F.avg_pool2d(pyramids[-1], kernel_size=2, stride=2)
        pyramids.append(downsampled)
    return pyramids


def laplacian_pyramid(gaussian_pyramids):
    """
    Generate a Laplacian pyramid from a Gaussian pyramid.
    """
    laplacian_pyramids = [gaussian_pyramids[-1]]
    for i in range(len(gaussian_pyramids) - 1, 0, -1):
        original_size = gaussian_pyramids[i-1].size()
        upsampled = F.interpolate(gaussian_pyramids[i], size=(original_size[2], original_size[3]), mode='nearest')
        laplacian = gaussian_pyramids[i-1] - upsampled
        laplacian_pyramids.append(laplacian)
    return laplacian_pyramids[::-1]

def reconstruct_pyramid(laplacian_pyramids):
    """
    Reconstruct an image from its Laplacian pyramid.
    """
    image = laplacian_pyramids[0]
    for i in range(1, len(laplacian_pyramids)):
        upsampled_size = laplacian_pyramids[i].size()
        image = F.interpolate(image, size=(upsampled_size[2], upsampled_size[3]), mode='nearest') + laplacian_pyramids[i]
    return image

def laplacian_pyramid_fusion(image1, image2, levels=4):
    """
    Fuse two images using Laplacian pyramid fusion.
    """
    # Convert images to PyTorch tensors if they're not already
    if not isinstance(image1, torch.Tensor):
        image1 = torch.tensor(image1, dtype=torch.float32)
    if not isinstance(image2, torch.Tensor):
        image2 = torch.tensor(image2, dtype=torch.float32)
    
    # Ensure images are 4D (batch_size, channels, height, width)
    if len(image1.shape) == 3:
        image1 = image1.unsqueeze(0)
    if len(image2.shape) == 3:
        image2 = image2.unsqueeze(0)
    
    gp1 = gaussian_pyramid(image1, levels)
    gp2 = gaussian_pyramid(image2, levels)
    
    lp1 = laplacian_pyramid(gp1)
    lp2 = laplacian_pyramid(gp2)
    
    # Fuse Laplacian pyramids
    lp_fused = [0.5 * (a + b) for a, b in zip(lp1, lp2)]
    
    # Reconstruct the fused image
    fused_image = reconstruct_pyramid(lp_fused)
    return fused_image.squeeze()  # Remove batch dimension



def increase_dataset():

    # Create `clean_images_by_digit` is dict with keys being digit classes (0-9) and values being lists of flattened clean images for that digit
    clean_images_by_digit = {}
    clean_images_by_digit_filenames = {}
    # Iterate over all clean images and group them by digit
    for filename in os.listdir('./Data/clean'):
        # Extract the digit from the filename
        _type, _no, number = filename.split('_')
        # remove the file extension
        number = number.split('.')[0]
        if number not in clean_images_by_digit:
            clean_images_by_digit[number] = []
            clean_images_by_digit_filenames[number] = []
        # use torchvision to load the image
        # clean_image = Image.open(os.path.join('/Users/jyotir/Desktop/DL/Assignments/A3/DLA3/Data/clean', filename)).convert('L')
        clean_image = torchvision.io.read_image(os.path.join('./Data/clean', filename))
        clean_image = transforms.Grayscale()(clean_image)

        # convert from torch.uint8 to torch.float
        clean_image = clean_image.float()
    

        clean_images_by_digit[number].append(np.array(clean_image).flatten())
        clean_images_by_digit_filenames[number].append(filename)

    

    # Create `augmented_images_by_digit` is dict with keys being digit classes (0-9) and values being lists of flattened augmented images for that digit
    augmented_images_by_digit = {}
    augmented_images_by_digit_filenames = {}
    # Iterate over all augmented images and group them by digit
    for filename in os.listdir('./Data/aug'):
        # Extract the digit from the filename
        _type, _no, number = filename.split('_')
        # remove the file extension
        number = number.split('.')[0]
        if number not in augmented_images_by_digit:
            augmented_images_by_digit[number] = []
            augmented_images_by_digit_filenames[number] = []
        # augmented_image = Image.open(os.path.join('/Users/jyotir/Desktop/DL/Assignments/A3/DLA3/Data/aug', filename)).convert('L')
        augmented_image = torchvision.io.read_image(os.path.join('./Data/aug', filename))
        augmented_image = transforms.Grayscale()(augmented_image)
        augmented_image = augmented_image.float()

        augmented_images_by_digit[number].append(np.array(augmented_image).flatten()) # Dictionary structure: {digit: [flattened_image1, flattened_image2, ...]}
        augmented_images_by_digit_filenames[number].append(filename)




    # Assuming augmented_images_by_digit contain flattened images
    all_augmented_images_data_aug = np.concatenate([np.array(images) for images in augmented_images_by_digit.values()], axis=0)
    pca_data_aug = PCA(n_components=0.95)
    pca_data_aug.fit(all_augmented_images_data_aug)
    # Transform augmented images
    augmented_images_pca_data_aug = {digit: pca_data_aug.transform(np.array(images)) for digit, images in augmented_images_by_digit.items()}


    # Initialize a dictionary to hold the new noisy images for each digit
    new_noisy_images = {digit: [] for digit in range(10)}

    for digit in range(10):
        # Step 1: Fit GMM to the PCA-reduced augmented images for the current digit
        n_components = 20
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(all_augmented_images_data_aug[digit].reshape(-1, 1)) # reshape is required because the input to fit() should be a 2D array
        
        # Step 2: Predict cluster labels for the augmented images
        cluster_labels = gmm.predict(all_augmented_images_data_aug[digit].reshape(-1, 1)) # reshape is required because the input to predict() should be a 2D array
        
        # Step 2 & 3: Pick two random images from two different clusters and fuse them
        unique_clusters = np.unique(cluster_labels)
        if len(unique_clusters) >= 2:
            # add 1000 new noisy images
            for i in range(1200):
                two_clusters = np.random.choice(unique_clusters, 2, replace=False)
                image_indices = [random.choice(np.where(cluster_labels == cluster)[0]) for cluster in two_clusters]
                two_images = [augmented_images_by_digit[str(digit)][idx] for idx in image_indices]

                # convert dimensions to 28x28 for two_images
                two_images = [image.reshape(28, 28) for image in two_images]                
                
                # Fuse the images
                # new_noisy_image = laplacian_pyramid_fusion(two_images[0], two_images[1], levels=1)
                if i<600:
                    new_noisy_image = fusion_with_noise(two_images[0], two_images[1], weight1=0.5, weight2=0.5, noise_level=5)
                else:
                    new_noisy_image = laplacian_pyramid_fusion(two_images[0], two_images[1], levels=1)

                new_noisy_images[digit].append(new_noisy_image)

        else:
            continue

    
        # save the new noisy images in 'aug' folder
    for digit, images in new_noisy_images.items():
        for i, image in enumerate(images):

            # if tensor then convert to numpy
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            # convert the numpy array to PIL image
            # new_noisy_image = Image.fromarray(image.astype(np.uint8))
            new_noisy_image = torch.tensor(image, dtype=torch.uint8)
            new_noisy_image = torchvision.transforms.ToPILImage()(new_noisy_image)
            
            # new_noisy_image = Image.fromarray(image.reshape(28, 28).astype(np.uint8))
            # new_noisy_image = Image.fromarray(image.astype(np.uint8))
            new_noisy_image.save(f'./Data/aug/newnoisy_{i}_{digit}.png')





def create_mappings():

    # Create `clean_images_by_digit` is dict with keys being digit classes (0-9) and values being lists of flattened clean images for that digit
    clean_images_by_digit = {}
    clean_images_by_digit_filenames = {}
    # Iterate over all clean images and group them by digit
    for filename in os.listdir('./Data/clean'):
        # Extract the digit from the filename
        _type, _no, number = filename.split('_')
        # remove the file extension
        number = number.split('.')[0]
        if number not in clean_images_by_digit:
            clean_images_by_digit[number] = []
            clean_images_by_digit_filenames[number] = []
        # clean_image = Image.open(os.path.join('/Users/jyotir/Desktop/DL/Assignments/A3/DLA3/Data/clean', filename)).convert('L')
        clean_image = torchvision.io.read_image(os.path.join('./Data/clean', filename))
        clean_image = transforms.Grayscale()(clean_image)
        clean_image = clean_image.float()

        clean_images_by_digit[number].append(np.array(clean_image).flatten())
        clean_images_by_digit_filenames[number].append(filename)


    # Create `augmented_images_by_digit` is dict with keys being digit classes (0-9) and values being lists of flattened augmented images for that digit
    augmented_images_by_digit = {}
    augmented_images_by_digit_filenames = {}
    # Iterate over all augmented images and group them by digit
    for filename in os.listdir('./Data/aug'):
        # Extract the digit from the filename
        _type, _no, number = filename.split('_')
        # remove the file extension
        number = number.split('.')[0]
        if number not in augmented_images_by_digit:
            augmented_images_by_digit[number] = []
            augmented_images_by_digit_filenames[number] = []
        # augmented_image = Image.open(os.path.join('/Users/jyotir/Desktop/DL/Assignments/A3/DLA3/Data/aug', filename)).convert('L')
        augmented_image = torchvision.io.read_image(os.path.join('./Data/aug', filename))
        augmented_image = transforms.Grayscale()(augmented_image)
        augmented_image = augmented_image.float()
        
        augmented_images_by_digit[number].append(np.array(augmented_image).flatten()) # Dictionary structure: {digit: [flattened_image1, flattened_image2, ...]}
        augmented_images_by_digit_filenames[number].append(filename)



    # Assuming clean_images_by_digit and augmented_images_by_digit contain flattened images
    all_clean_images = np.concatenate([np.array(images) for images in clean_images_by_digit.values()], axis=0)
    all_augmented_images = np.concatenate([np.array(images) for images in augmented_images_by_digit.values()], axis=0)
    # Combine all images for PCA fitting
    all_images = np.concatenate([all_clean_images, all_augmented_images], axis=0)


    pca = PCA(n_components=0.95)
    pca.fit(all_images)
    # Transform both clean and augmented images
    clean_images_pca = {digit: pca.transform(np.array(images)) for digit, images in clean_images_by_digit.items()}
    augmented_images_pca = {digit: pca.transform(np.array(images)) for digit, images in augmented_images_by_digit.items()}



    count = 0
    gmm_models = {}
    mappings = {digit: {} for digit in range(10)}  # Prepare mappings
    cluster_representatives = {}  # Store the clean image with the highest score for each cluster
    for digit, clean_images in clean_images_pca.items():
        n_components = 200
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        
        # Fit the GMM to clean images
        gmm.fit(clean_images)
        clean_cluster_labels = gmm.predict(clean_images) # Get the cluster labels for clean images
        scores = gmm.predict_proba(clean_images) # Get the scores for clean images
        
        gmm_models[digit] = gmm
        # Identify the clean image with the highest score for each cluster
        cluster_representatives[digit] = {}
        for cluster_idx in range(n_components):
            cluster_image_scores = scores[:, cluster_idx] # Get the scores for the cluster
            best_clean_image_idx = np.argmax(cluster_image_scores) # Get the index of the clean image with the highest score in the cluster
            best_clean_path = clean_images_by_digit_filenames[digit][best_clean_image_idx] # Get the path of the clean image with the highest score in the cluster
            cluster_representatives[digit][cluster_idx] = best_clean_path


        for idx, aug_img_data in enumerate(augmented_images_by_digit[digit]):

            # apply PCA to the augmented image
            aug_img_data = pca.transform(aug_img_data.reshape(1, -1)).squeeze() # reshape (1, -1) to make it 2D if it's 1D and if it's 2D, it won't change anything, then squeeze to remove the first dimension

            scores = gmm.predict_proba(aug_img_data.reshape(1, -1))
            best_match_idx = np.argmax(scores)
            # Map the augmented image to the clean image with the highest score in the cluster
            aug_img_path = augmented_images_by_digit_filenames[digit][idx]
            mappings[int(digit)][aug_img_path] = cluster_representatives[digit][best_match_idx]

        # print(f"Digit {digit} done")

    return mappings



class ToFloatAndNormalize(torch.nn.Module):
    """
    Transform to convert image tensor to float and normalize it.
    """
    def __init__(self, mean=0.5, std=0.5):
        super(ToFloatAndNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        # Ensure x is a float tensor
        x = x.float() / 255.0
        # If mean and std are tuples, convert them to tensors with matching device and dtype
        if isinstance(self.mean, tuple):
            mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.tensor(self.std, dtype=x.dtype, device=x.device)
        else:
            mean = self.mean
            std = self.std
        # Subtract mean and divide by std
        return (x - mean) / std





class AlteredMNIST:
    """
    dataset description:
    
    X_I_L.png
    X: {aug=[augmented], clean=[clean]}
    I: {Index range(0,60000)}
    L: {Labels range(10)}
    
    Write code to load Dataset
    """
    def __init__(self):

        increase_dataset()
        self.augmented_dir = './Data/aug'
        self.augmented_filenames = os.listdir(self.augmented_dir)
        self.clean_dir = './Data/clean'
        # self.transform_data = transforms.Compose([transforms.Lambda(lambda x: x.float() / 255.0),
        #                                           transforms.Normalize((0.5,), (0.5,))
        #                                           ])
        self.transform_data = transforms.Compose([
            transforms.Grayscale(),
            ToFloatAndNormalize((0.5,), (0.5,))
            ])
        

        self.clean_images_by_number = self._load_clean_images_by_number(self.clean_dir)
        self.mappings = create_mappings()

    def _load_clean_images_by_number(self, clean_dir):
        clean_filenames = os.listdir(clean_dir)
        clean_images_by_number = {}
        for filename in clean_filenames:
            # Extract the number (Y value) from the filename
            _type, _no, number = filename.split('_')
            # remove the file extension
            number = number.split('.')[0]
            if number not in clean_images_by_number:
                clean_images_by_number[number] = []
            clean_images_by_number[number].append(os.path.join(clean_dir, filename))
        return clean_images_by_number

    def __len__(self):
        return len(self.augmented_filenames)

    def __getitem__(self, idx):
        aug_filename = self.augmented_filenames[idx]
        # Extract the Y value from the augmented image filename
        _type, _no, digit = aug_filename.split('_')
        y_value = digit.split('.')[0] # Remove the file extension

        mapping_digit = self.mappings[int(y_value)]
        clean_image_path = mapping_digit[aug_filename]

        # aug_image = Image.open(os.path.join(self.augmented_dir, aug_filename)).convert('L')
        aug_image = torchvision.io.read_image(os.path.join(self.augmented_dir, aug_filename))
        aug_image = transforms.Grayscale()(aug_image)

        # clean_image = Image.open(os.path.join(self.clean_dir, clean_image_path)).convert('L')
        clean_image = torchvision.io.read_image(os.path.join(self.clean_dir, clean_image_path))
        clean_image = transforms.Grayscale()(clean_image)
        

        if self.transform_data is not None:
            aug_image = self.transform_data(aug_image)
            clean_image = self.transform_data(clean_image)


        return aug_image, clean_image, torch.tensor(int(y_value))


def split_dataset(augmented_dir, split_ratios=(0.7, 0.15, 0.15)):
    augmented_filenames = os.listdir(augmented_dir)
    random.shuffle(augmented_filenames) # Shuffle the filenames to ensure randomness
    
    total_count = len(augmented_filenames)
    train_count = int(total_count * split_ratios[0])
    val_count = int(total_count * split_ratios[1])
    
    train_filenames = augmented_filenames[:train_count]
    val_filenames = augmented_filenames[train_count:train_count + val_count]
    test_filenames = augmented_filenames[train_count + val_count:]
    
    return train_filenames, val_filenames, test_filenames



####----------------------------------------AlteredMNIST End----------------------------------------####


####----------------------------------------ResBlock Start----------------------------------------####

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
    
####----------------------------------------ResBlock End----------------------------------------####
    

####----------------------------------------Q1 Model Start----------------------------------------####

class Encoder_AE(nn.Module):
    def __init__(self):
        super(Encoder_AE, self).__init__()
        self.initial = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResBlock(32, 64, stride=2)
        self.resblock2 = ResBlock(64, 128, stride=2)
        self.resblock3 = ResBlock(128, 128, stride=2)  # Preserves channel depth, adjusts for spatial reduction

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        return x

class Decoder_AE(nn.Module):
    def __init__(self):
        super(Decoder_AE, self).__init__()
        self.resblock1 = ResBlock(128, 64)
        # Start with a reduction in channels, no spatial change yet
        self.resblock2 = ResBlock(64, 32)
        # Further reduction in channels
        self.resblock3 = ResBlock(32, 16)
        # Final reduction to match the initial channel depth
        self.final = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        # Final convolution to produce output with desired channel

    def forward(self, x):
        x = self.resblock1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # First upsampling
        x = self.resblock2(x)
        x = F.interpolate(x, size=(28, 28), mode='nearest')  # Directly adjust to target size
        x = self.resblock3(x)
        x = self.final(x)
        return x

####----------------------------------------Q1 Model End----------------------------------------####
    
####----------------------------------------Q2 Model Start----------------------------------------####

class Encoder_VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super(Encoder_VAE, self).__init__()
        self.initial = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.resblocks = nn.Sequential(
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 512, stride=2)  # Additional deeper layers
        )
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)  # Adjusted for additional depth
        self.fc_var = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.resblocks(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder_VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(Decoder_VAE, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)  # Start from a compact representation

        # Correctly configure transposed convolutions for upsampling
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock1 = ResBlock(256, 256)  # Use ResBlocks to refine features after each upsampling

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock2 = ResBlock(128, 128)

        # Additional upsampling and processing layers
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock3 = ResBlock(64, 64)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock4 = ResBlock(32, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)  # Final layer to get to 1 channel output

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 2, 2)  # Start with a small spatial dimension and large channel dimension

        x = self.upconv1(x)
        x = F.relu(x)
        x = self.resblock1(x)

        x = self.upconv2(x)
        x = F.relu(x)
        x = self.resblock2(x)

        x = self.upconv3(x)
        x = F.relu(x)
        x = self.resblock3(x)

        x = self.upconv4(x)
        x = F.relu(x)
        x = self.resblock4(x)

        x = self.final(x)

        # Explicitly resize the output to match the desired dimensions (28x28)
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False) # mode='bilinear' is used for resizing, 'bilinear' refers to bilinear interpolation
        # align_corners=False is used to ensure the output matches the desired size exactly
        
        return x
    
####----------------------------------------Q2 Model End----------------------------------------####

####----------------------------------------Encoder/Decoder Assignment Start----------------------------------------####

class Encoder:
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """

    def __init__(self):
        self.encoder = Encoder_AE().to(DEVICE)

    def assign_encoder(self, loss_fn):
        ### Check instance of loss_fn
        if isinstance(loss_fn, AELossFn):
            self.encoder = Encoder_AE().to(DEVICE)
        elif isinstance(loss_fn, VAELossFn):
            self.encoder = Encoder_VAE().to(DEVICE)
        elif isinstance(loss_fn, CVAELossFn):
            self.encoder = Encoder_CVAE().to(DEVICE)
    

class Decoder:
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """

    def __init__(self):
        self.decoder = Decoder_AE().to(DEVICE)


    def assign_decoder(self, loss_fn):
        ### Check instance of loss_fn
        if isinstance(loss_fn, AELossFn):
            self.decoder = Decoder_AE().to(DEVICE)
        elif isinstance(loss_fn, VAELossFn):
            self.decoder = Decoder_VAE().to(DEVICE)
        elif isinstance(loss_fn, CVAELossFn):
            self.decoder = Decoder_CVAE().to(DEVICE)


####----------------------------------------Encoder/Decoder Assignment End----------------------------------------####


####----------------------------------------Loss Function Start----------------------------------------####

class AELossFn:
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self):
        super(AELossFn, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def __call__(self, outputs, targets): #__call__ method is called when the instance is called as a function
        loss = self.criterion(outputs, targets)
        return loss

class VAELossFn:
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    def __init__(self):
        super(VAELossFn, self).__init__()


    def __call__(self, recon_x, x, mu, log_var): #__call__ method is called when the instance is called as a function
        # Calculate the reconstruction loss
        mse = F.mse_loss(recon_x, x, reduction='sum')
        # Calculate the KL divergence
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return mse + kld
    
####----------------------------------------Loss Function End----------------------------------------####


def ParameterSelector(E, D):
    """
    Write code for selecting parameters to train
    """
    encoder_params = list(E.encoder.parameters())
    decoder_params = list(D.decoder.parameters())
    
    # Combine the parameter lists
    combined_params = encoder_params + decoder_params
    
    return combined_params



####----------------------------------------Trainer Start----------------------------------------####

def calculate_metrics_inbuilt(reconstructed, original):
        
        # if reconstructed is a tensor that requires grad, detach it
        if reconstructed.requires_grad:
            reconstructed = reconstructed.detach()
        if original.requires_grad:
            original = original.detach()
        

        # Assuming images are single-channel and on CPU for skimage compatibility
        # Ensure image tensor is on CPU and convert to numpy for SSIM
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        original_np = original.squeeze().cpu().numpy()
        
        psnr_value = peak_signal_to_noise_ratio_torch(reconstructed, original)
        ssim_value = calculate_ssim(original_np, reconstructed_np, data_range = reconstructed_np.max() - reconstructed_np.min())

        return psnr_value, ssim_value

def peak_signal_to_noise_ratio_torch(img1, img2):
    mse = F.mse_loss(img1, img2)
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


class AETrainer:
    """
    Write code for training AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, device='cpu'):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.DEVICE = device
        self.dataloader = dataloader

        # call the train method to start training
        self.train(dataloader)


    def train(self, train_loader, n_epochs=50):
        n_epochs = 50
        for epoch in range(1, n_epochs+1):
            epoch_loss = 0.0
            minibatch = 0

            total_psnr = 0.0
            total_ssim = 0.0
            total_images = 0

            for augmented_images, clean_images, _ in train_loader:
                
                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_images = 0

                self.optimizer.zero_grad()

                # Forward pass
                augmented_images = augmented_images.to(DEVICE)
                clean_images = clean_images.to(DEVICE)
                outputs = self.encoder.encoder(augmented_images)
                outputs = self.decoder.decoder(outputs)

                
                loss = self.loss_fn(outputs, clean_images)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                minibatch += 1


                # for each 10th minibatch use only this print statement
                if minibatch % 10 == 0:
                    # Calculate metrics for the batch
                    for original, reconstructed in zip(clean_images, outputs):
                        psnr, ssim = calculate_metrics_inbuilt(reconstructed, original)
                        batch_psnr += psnr
                        batch_ssim += ssim
                        batch_images += 1

                    batch_psnr /= batch_images
                    batch_ssim /= batch_images

                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,batch_ssim))


            # Calculate metrics for the epoch
            for original, reconstructed in zip(clean_images, outputs):
                psnr, ssim = calculate_metrics_inbuilt(reconstructed, original)
                total_psnr += psnr
                total_ssim += ssim
                total_images += 1

            total_psnr /= total_images
            total_ssim /= total_images

            loss = epoch_loss / len(train_loader)

            if epoch % 10 == 0:
                # plot the t-SNE plot
                all_embeddings = self.collect_embeddings(self.dataloader)
                self.plot_tsne(epoch, all_embeddings)

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,loss,total_ssim))

            if epoch == 50:
                # Save the model after the final epoch without using state_dict
                torch.save(self.encoder, "q1_encoder_final.pt")
                torch.save(self.decoder, "q1_decoder_final.pt")
                # break

    def collect_embeddings(self, dataloader):
        
        self.encoder.encoder.eval()  # Ensure the encoder is in evaluation mode
        embeddings = []
        with torch.no_grad():  # No need to compute gradients
            for augmented_images, _, _ in dataloader:
                augmented_images = augmented_images.to(DEVICE)
                outputs = self.encoder.encoder(augmented_images)
                # Ensure outputs are flattened
                embeddings.append(outputs.view(outputs.size(0), -1).cpu().numpy())
        
        self.encoder.encoder.train()  # Set the encoder back to training mode
        return np.concatenate(embeddings, axis=0)  # Concatenate all batch-wise embeddings into a 2D array


    def plot_tsne(self, epoch, all_embeddings):
        """
        Reduce embeddings to 3D using t-SNE and plot.
        """
        tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300) # verbose = 1 to see the progress
        
        tsne_results = tsne.fit_transform(all_embeddings)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], cmap='Spectral')
        plt.title("3D TSNE Plot")
        plt.savefig(f"AE_epoch_{epoch}.png")
        plt.close() # Close the plot to prevent display in Jupyter


class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer, device='cpu'):
        self.encoder = Encoder_VAE().to(DEVICE)
        self.decoder = Decoder_VAE().to(DEVICE)
        # reinstantiate the optimizer
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-3)
        self.loss_fn = VAELossFn()
        self.DEVICE = device
        self.dataloader = dataloader

        # call the train method to start training
        self.train(dataloader)


    def train(self, train_loader, n_epochs=50):
        n_epochs = 50
        for epoch in range(1, n_epochs+1):
            
            epoch_loss = 0.0

            total_psnr = 0.0
            total_ssim = 0.0
            total_images = 0

            minibatch = 0

            for augmented_images, clean_images, _ in train_loader:

                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_images = 0
                minibatch += 1


                self.optimizer.zero_grad()

                augmented_images = augmented_images.to(DEVICE)
                clean_images = clean_images.to(DEVICE)

                # Forward pass
                mu, log_var = self.encoder.forward(augmented_images)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std
                outputs = self.decoder.forward(z)

                # outputs, mu, log_var = model(augmented_images)
                loss = self.loss_fn(outputs, clean_images, mu, log_var)


                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() 



                # for each 10th minibatch use only this print statement
                if minibatch % 10 == 0:
                    # Calculate metrics for the batch
                    for original, reconstructed in zip(clean_images, outputs):
                        psnr, ssim = calculate_metrics_inbuilt(reconstructed, original)
                        batch_psnr += psnr
                        batch_ssim += ssim
                        batch_images += 1

                    batch_psnr /= batch_images
                    batch_ssim /= batch_images

                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,batch_ssim))

            
            # Calculate metrics for the epoch
            for original, reconstructed in zip(clean_images, outputs):
                psnr, ssim = calculate_metrics_inbuilt(reconstructed, original)
                total_psnr += psnr
                total_ssim += ssim
                total_images += 1

            total_psnr /= total_images
            total_ssim /= total_images


            epoch_loss /= len(train_loader)
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch,epoch_loss,total_ssim))


            if epoch % 10 == 0:
                # plot the t-SNE plot
                all_embeddings = self.collect_embeddings(self.dataloader)
                self.plot_tsne(epoch, all_embeddings)

            if epoch == 50:
                # Save the model after the final epoch without using state_dict
                torch.save(self.encoder, "q2_encoder_final.pt")
                torch.save(self.decoder, "q2_decoder_final.pt")
                # break



    def collect_embeddings(self, dataloader):
        
        self.encoder.eval()  # Ensure the encoder is in evaluation mode
        embeddings = []
        with torch.no_grad():  # No need to compute gradients
            for augmented_images, _, _ in dataloader:
                augmented_images = augmented_images.to(DEVICE)
                # outputs = self.encoder(augmented_images)

                mu, log_var = self.encoder.forward(augmented_images)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std

                # Ensure outputs are flattened
                embeddings.append(z.cpu().numpy())
        
        self.encoder.train()  # Set the encoder back to training mode
        return np.concatenate(embeddings, axis=0)  # Concatenate all batch-wise embeddings into a 2D array


    def plot_tsne(self, epoch, all_embeddings):
        """
        Reduce embeddings to 3D using t-SNE and plot.
        """
        tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300) # verbose = 1 to see the progress
        
        tsne_results = tsne.fit_transform(all_embeddings)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], cmap='Spectral')
        plt.title("3D TSNE Plot")
        plt.savefig(f"VAE_epoch_{epoch}.png")
        plt.close() # Close the plot to prevent display in Jupyter


####----------------------------------------Trainer End----------------------------------------####

class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self,gpu=False):
        if(gpu):
            self.encoder=torch.load('./q1_encoder_final.pt',map_location='cuda')
            self.decoder=torch.load('./q1_decoder_final.pt',map_location='cuda')
            self.device='cuda'
        else:
            self.encoder=torch.load('./q1_encoder_final.pt',map_location='cpu')
            self.decoder=torch.load('./q1_decoder_final.pt',map_location='cpu')
            self.device='cpu'
        self.encoder.encoder.eval()
        self.decoder.decoder.eval()

        self.transform_data = transforms.Compose([
            transforms.Grayscale(),
            ToFloatAndNormalize((0.5,), (0.5,))
            ])
        
    def calculate_metrics_inbuilt(self, reconstructed, original):
        
        # if reconstructed is a tensor that requires grad, detach it
        if reconstructed.requires_grad:
            reconstructed = reconstructed.detach()
        if original.requires_grad:
            original = original.detach()
        

        # Assuming images are single-channel and on CPU for skimage compatibility
        # Ensure image tensor is on CPU and convert to numpy for SSIM
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        original_np = original.squeeze().cpu().numpy()
        
        psnr_value = peak_signal_to_noise_ratio_torch(reconstructed, original)
        ssim_value = calculate_ssim(original_np, reconstructed_np, data_range = reconstructed_np.max() - reconstructed_np.min())

        return psnr_value, ssim_value

    def from_path(self, sample, original, type):
        
        "Compute similarity score of both 'sample' and 'original' and return in float"
        augmented_image = torchvision.io.read_image(sample)
        augmented_image = transforms.Grayscale()(augmented_image)

        clean_image = torchvision.io.read_image(original)
        clean_image = transforms.Grayscale()(clean_image)

        if self.transform_data is not None:
            augmented_image = self.transform_data(augmented_image)
            clean_image = self.transform_data(clean_image)

        augmented_image = augmented_image.to(self.device)
        clean_image = clean_image.to(self.device)

        with torch.no_grad():
            # increase the dimensions of the image tensor to match the expected input shape
            augmented_image = augmented_image.unsqueeze(0)
            outputs = self.encoder.encoder(augmented_image)
            outputs = self.decoder.decoder(outputs)

        psnr, ssim = self.calculate_metrics_inbuilt(outputs, clean_image)

        if type == 'psnr':
            return psnr
        elif type == 'ssim':
            return ssim
        

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self,gpu=False):
        if(gpu):
            self.encoder=torch.load('./q2_encoder_final.pt',map_location='cuda')
            self.decoder=torch.load('./q2_decoder_final.pt',map_location='cuda')
            self.device='cuda'
        else:
            self.encoder=torch.load('./q2_encoder_final.pt',map_location='cpu')
            self.decoder=torch.load('./q2_decoder_final.pt',map_location='cpu')
            self.device='cpu'
        self.encoder.eval()
        self.decoder.eval()

        self.transform_data = transforms.Compose([
            transforms.Grayscale(),
            ToFloatAndNormalize((0.5,), (0.5,))
            ])
        
    def calculate_metrics_inbuilt(self, reconstructed, original):
        
        # if reconstructed is a tensor that requires grad, detach it
        if reconstructed.requires_grad:
            reconstructed = reconstructed.detach()
        if original.requires_grad:
            original = original.detach()
        

        # Assuming images are single-channel and on CPU for skimage compatibility
        # Ensure image tensor is on CPU and convert to numpy for SSIM
        reconstructed_np = reconstructed.squeeze().cpu().numpy()
        original_np = original.squeeze().cpu().numpy()
        
        psnr_value = peak_signal_to_noise_ratio_torch(reconstructed, original)
        ssim_value = calculate_ssim(original_np, reconstructed_np, data_range = reconstructed_np.max() - reconstructed_np.min())

        return psnr_value, ssim_value

    def from_path(self, sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        augmented_image = torchvision.io.read_image(sample)
        augmented_image = transforms.Grayscale()(augmented_image)

        clean_image = torchvision.io.read_image(original)
        clean_image = transforms.Grayscale()(clean_image)

        if self.transform_data is not None:
            augmented_image = self.transform_data(augmented_image)
            clean_image = self.transform_data(clean_image)

        augmented_image = augmented_image.to(self.device)
        clean_image = clean_image.to(self.device)

        with torch.no_grad():
            # increase the dimensions of the image tensor to match the expected input shape
            augmented_image = augmented_image.unsqueeze(0)
            mu, log_var = self.encoder.forward(augmented_image)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            outputs = self.decoder.forward(z)

        psnr, ssim = self.calculate_metrics_inbuilt(outputs, clean_image)

        if type == 'psnr':
            return psnr
        elif type == 'ssim':
            return ssim


# class CVAELossFn():
#     """
#     Write code for loss function for training Conditional Variational AutoEncoder
#     """
#     pass

# class CVAE_Trainer:
#     """
#     Write code for training Conditional Variational AutoEncoder here.
    
#     for each 10th minibatch use only this print statement
#     print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
#     for each epoch use only this print statement
#     print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
#     After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
#     """
#     pass

# class CVAE_Generator:
#     """
#     Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
#     use forward pass of both encoder-decoder to get output image conditioned to the class.
#     """
    
#     def save_image(digit, save_path):
#         pass


def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()

def structure_similarity_index(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    # Constants
    window_size, channels = 11, 1
    K1, K2, DR = 0.01, 0.03, 255
    C1, C2 = (K1*DR)**2, (K2*DR)**2

    window = torch.randn(11)
    window = window.div(window.sum())
    window = window.unsqueeze(1).mul(window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    mu12 = mu1.pow(2).mul(mu2.pow(2))

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channels) - mu1.pow(2)
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channels) - mu2.pow(2)
    sigma12 =  F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu12


    SSIM_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - SSIM_n / (denom + 1e-8)), min=0.0, max=1.0).mean().item()


    

class CVAE_Generator:
    def __init__(self,gpu=False):
        if(gpu):
            self.encoder=torch.load('encoder_CVAE',map_location='cuda')
            self.decoder=torch.load('decoder_CVAE',map_location='cuda')
            self.device='cuda'
        else:
            self.encoder=torch.load('encoder_CVAE',map_location='cpu')
            self.decoder=torch.load('decoder_CVAE',map_location='cpu')
            self.device='cpu'
        self.encoder.eval()
        self.decoder.eval()
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,))  # Adjust these values based on your dataset normalization
        ])
    
    def save_image(self,digit, save_path):
        digit_tensor = torch.tensor(int(digit), dtype=torch.long)
        z = torch.randn(1,128)
        digit_tensor = digit_tensor.unsqueeze(0)

        decoded_image = self.decoder(z, digit_tensor)

        ### Display the generated image
        plt.imshow(decoded_image.squeeze().detach().cpu().numpy(), cmap='gray')




class CVAE_Trainer:
    def __init__(self, dataloader, encoder, decoder, loss_fn, optimizer):
        encoder.assign_encoder(loss_fn)
        decoder.assign_decoder(loss_fn)
        self.encoder = encoder.encoder
        self.decoder = decoder.decoder
        self.loss_fn = loss_fn
        self.loss_fn = CVAELossFn()
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.optimizer = torch.optim.Adam(ParameterSelector(encoder, decoder), lr=0.001)
        self.epochs = 10
        self.train()

    def train(self):
        prev_val_ssim=0
        n_epochs = 50
        for epoch in range(1, n_epochs + 1):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0.0
            total_train_ssim=0.0
            embeddings=[]
            labels=[]
            total_ssim = 0
            total_psnr = 0
            total_images = 0
            for i,(noisy_images, clean_images, digits) in enumerate(self.dataloader):
                noisy_images, clean_images, digits= noisy_images.to(DEVICE), clean_images.to(DEVICE), digits.to(DEVICE)
                
                self.optimizer.zero_grad()
                
                mu, logvar = self.encoder(noisy_images)
                
                z = self.reparameterize(mu, logvar)
                
                decoded_images = self.decoder(z, digits)
                if epoch % 10 == 0:
                    embeddings.extend(mu.view(mu.size(0), -1).cpu().detach().numpy())
                    labels.extend(digits.cpu().numpy())
                loss = self.loss_fn(decoded_images, clean_images, mu, logvar)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                # batch_train_scores = [calculate_image_metrics(decoded_images[j].detach().cpu(), clean_images[j].cpu()) for j in range(clean_images.size(0))]
                # batch_train_psnr, batch_train_ssim = zip(*batch_train_scores)

                # total_train_ssim += np.mean(batch_train_ssim)
                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_images = 0
                
                if i % 10 == 0:
                    for original, reconstructed in zip(clean_images, decoded_images):
                        psnr, ssim = calculate_metrics_inbuilt(reconstructed, original)
                        batch_psnr += psnr
                        batch_ssim += ssim
                        batch_images += 1
                    batch_psnr /= batch_images
                    batch_ssim /= batch_images
                    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,i,loss,batch_ssim))

            avg_loss = total_loss / len(self.dataloader)
            avg_train_ssim = total_train_ssim / len(self.dataloader)

            for original, reconstructed in zip(clean_images, decoded_images):
                psnr, ssim = calculate_metrics_inbuilt(reconstructed, original)
                total_psnr += psnr
                total_ssim += ssim
                total_images += 1

            total_psnr /= total_images
            total_ssim /= total_images

            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch, avg_loss, total_ssim))
            if epoch % 10 == 0:
                self.plot_tsne(np.array(embeddings), np.array(labels),epoch)

            if epoch == 50:
                checkpoint_path = 'encoder_CVAE.pt'
                torch.save(self.encoder, checkpoint_path)
                checkpoint_path = 'decoder_CVAE.pt'
                torch.save(self.decoder, checkpoint_path)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def plot_tsne(self, embeddings, labels, epoch):
        tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(embeddings)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=labels, cmap='tab10')
        plt.title(f"3D TSNE - Epoch {epoch}")
        plt.savefig(f"CVAE_epoch_{epoch}.png")
        plt.close()

    
class CVAELossFn(nn.Module):
    def forward(self, recon_x, x, mu, logvar):

        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # Calculate the KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss

        loss = recon_loss + kl_div
        
        return loss
    



class Encoder_CVAE(nn.Module):
    def __init__(self):
        super(Encoder_CVAE, self).__init__()
        in_channels = 1
        latent_dim = 128
        
        self.initial = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        # Removed the second last ResBlock
        self.resblocks = nn.Sequential(
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2)  # Last ResBlock now
        )
        # Adjusting for the output of the last ResBlock being 128 channels with spatial dimensions reduced
        self.fc_mu = nn.Linear(128 * 7 * 7, latent_dim)  # Adjusted for the new output size
        self.fc_var = nn.Linear(128 * 7 * 7, latent_dim)  # Adjusted for the new output size

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.resblocks(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var



    
class Decoder_CVAE(nn.Module):
    def __init__(self):
        super(Decoder_CVAE, self).__init__()
        latent_dim = 128
        num_classes = 10

        # Adjusting for removing a layer, now starting from 128 * 7 * 7 to match encoder's adjusted last layer output
        self.fc = nn.Linear(latent_dim + 128, 128 * 7 * 7)  # Correctly adjusted size
        # Adjusting the upconv layers to start directly from 128 channels
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock1 = ResBlock(64, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.resblock2 = ResBlock(32, 32)  # This becomes the new final upconv and resblock pair
        self.final = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.condition_embedding = nn.Embedding(num_classes, latent_dim)

    def forward(self, z, labels):
        labels_embedded = self.condition_embedding(labels)
        z_and_labels = torch.cat([z, labels_embedded], dim=1)
        x = self.fc(z_and_labels)
        x = x.view(-1, 128, 7, 7)  # Correctly adjusted size
        x = F.relu(self.upconv1(x))  # Adjusted to be the first upconv in the simplified decoder
        x = self.resblock1(x)
        x = F.relu(self.upconv2(x))
        x = self.resblock2(x)
        x = self.final(x)
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        return x



    
class ResBlock_CVAE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_CVAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

























