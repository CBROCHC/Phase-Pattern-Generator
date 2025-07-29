import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import torch
import time


def autodiff_phase_retrival(sourceA, targetA, epsilon_crit = 1e-1,max_iter = 500 ,lr = 0.1) -> (torch, list, np.matrix):
    """""
    This funtion implements a phase retrieval algorithm using autodifferentiation and gradient descent with PyTorch.
    The goal is to find an optimal phase that, when combined with a known amplitude in the modulator plane, 
    produces an image whose intensity is as similar as possible to a target image.
    
    arg:
        sourceA: known amplitude of the modulator plane (hologram)
        targetA: target amplitude (reconstruction)
        epsilon_crit: error threshold to stop optimization.
        max_iter
        lr: learing rate for the ADAM optimizer
    return:
        phase: optimized phase
        errors: list of error per iteration
        int (np.square(np.abs(U))): resulting intensity
    """""

    # Random phase initialization
    phase = np.random.uniform(-np.pi, np.pi, size=targetA.shape)
    phase = torch.tensor(phase, requires_grad=True)

    optimizer = torch.optim.Adam([phase], lr=lr) # ADAM optimizer to phase optimization

    errors = []
    t = 0
    loss = torch.inf

    while loss > epsilon_crit and t < max_iter:
        optimizer.zero_grad()

        # Complex field: sourceA * exp(i*phi); Fourier plane, random phase (at the round 1)
        u = sourceA * torch.exp(1j * phase)

        # Complex field: Fourier plane   FFT→   Complex field: image (real) plane;
        # we apply the module to obtain the amplitude
        U = torch.abs(torch.fft.fftshift(torch.fft.fft2(u)))

        # calculation of the cost function (the distance from the target intensity)
        loss = torch.sqrt(torch.sum(torch.pow(U - targetA, 2)))
        print(f"Iteración {t} - Error = {loss:.4e}")
        loss.backward() # Gradient Calculation
        optimizer.step() # Phase update (depending on the error)

        errors.append(float(loss))
        t +=1


    with torch.no_grad():
       return phase, errors, np.square(np.abs(U))  # Get the resulting intensity with the target phase and the errors



if __name__ == '__main__':
#---------------------------------------------------------------------------------------
#   image path or intensity measure, is converted to a grayscale image with values between 0 and 255 (intensity measure)
    image_path = 'C:\\Users\\Icaro\\PycharmProjects\\Phase_Recovery\\sakura.jpg'
    img = Image.open(image_path).convert('L')

    img = img.resize((1280, 1024)) # Resolution adjustment, up to a maximum of 1024x1280 (HxW) (the size of the SLM)
#----------------------------------------------------------------------------------------

    target_int = np.array(img, dtype=np.float32)
    target_amplitude = np.sqrt(target_int)

    e = np.sum(target_amplitude) # correction to normalize data when plotting

    target_amplitude = target_amplitude / target_amplitude.max() # Normalize amplitude [0,1]

    #target_amplitude = target_amplitude.max() - target_amplitude #invert the amplitude values

    # Padding to fit SLM size
    height, width = target_amplitude.shape
    print("Original shape: ",target_amplitude.shape)
    top = (1024 - height) // 2
    bottom = (1024 - height) - top
    left = (1280 - width) // 2
    right = (1280 - width) - left

    target_amplitude = np.pad(target_amplitude, ((top, bottom), (left, right)), 'constant')
    print("After padding: ",target_amplitude.shape)

    target_amplitude = torch.tensor(target_amplitude, dtype=torch.float32) # np matrix to tensor conversion

# We take a fixed amplitude that will be used to build the field with the phase to be optimized
    source_amplitude = torch.ones_like(target_amplitude, dtype=torch.float32)
    source_amplitude = source_amplitude / e

#-----------------------------------------------------------------------------------
    inicio = time.perf_counter()
    # call to the phase acquisition and optimization function
    recovered_phase, errors, int = autodiff_phase_retrival(source_amplitude, target_amplitude, epsilon_crit=1e-1, max_iter=50, lr = 0.05)

    final = time.perf_counter()
    print(f"Tiempo transcurrido: {(final - inicio)} segundos")

    recovered_phase = recovered_phase.detach().numpy()
    recovered_phase = np.mod(recovered_phase, 2*np.pi)

#--------------------------------------------------------------
# Storing the resulting phase in an h5 file
    with h5py.File('cross_Autodiff.h5', 'w') as f:
        f.create_dataset('fase', data=recovered_phase)
        f.create_dataset('imagen', data=target_amplitude)
#--------------------------------------------------------------
# Show results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title("Target Intensity")
    plt.imshow(target_amplitude**2, cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("Recovered Phase")
    plt.imshow(recovered_phase,vmin=-np.pi, vmax=np.pi)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 4, 3)
    plt.title("Reconstructed Intensity")
    plt.imshow(int, cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 4, 4)
    plt.title("Error")
    plt.plot(errors)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.show()