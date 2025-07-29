import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import h5py


def gerchberg_saxton(source_amp, target_amp, epsilon_crit=1e-6, max_iter=500):
    """
    source_amp: Amplitud del plano del modulador (holograma)
    target_amp: Amplitud objetivo (reconstrucción)
    """

    phase = 2*np.pi*np.random.rand(1024,1280)-np.pi
    u=np.zeros((1024,1280),dtype=complex)

    epsilon_cur = np.inf
    errors = []
    t = 0

    while epsilon_cur > epsilon_crit and t < max_iter:
        # Fourier plane, random phase (at the round 1) and gaussian beam
        u = target_amp * np.exp(1j * phase)
        # To the real plane...
        u = np.fft.fft2(u)
        u = np.fft.fftshift(u)
        # Calculate the intensity
        int = np.square(np.abs(u))
        # Let's renormalize the intensity in the range [0,1]
        std_int = int / int.max()
        # What's the distance from the target intensity?
        errors.append(np.sum(std_int - target_amplitude ** 2) ** 2)
        phase = np.angle(u)
        loss = np.sum(std_int - target_amplitude**2) ** 2
        errors.append(loss)
        print(f"Iteración {t} - Error = {loss:.4e}")
        u = target_amplitude * np.exp(1j * phase)
        # Back to our fourier plane
        u = np.fft.ifftshift(u)
        u = np.fft.ifft2(u)
        # The phase that we need to imprint by the SLM is :
        phase = np.angle(u)

        t += 1

    # Retorna la fase a aplicar en el modulador
    return phase, errors, np.square(np.abs(u))

if __name__ == '__main__':   #TODO ajustar los shifts
    image_path = 'C:\\Users\\Icaro\\PycharmProjects\\Phase_Recovery\\hpk.jpg'
    img = Image.open(image_path).convert('L')
    img = img.resize((1280, 1024))

    target_int = np.array(img, dtype=np.float32)
    target_amplitude = np.sqrt(target_int)
    #target_amplitude = target_amplitude.max() - target_amplitude

    height, width = target_amplitude.shape
    print("Original shape: ", target_amplitude.shape)
    top = (1024 - height) // 2
    bottom = (1024 - height) - top
    left = (1280 - width) // 2
    right = (1280 - width) - left

    target_amplitude = np.pad(target_amplitude, ((top, bottom), (left, right)), 'constant')
    print("After padding: ", target_amplitude.shape)

    e = np.sum(target_amplitude)
    target_amplitude = target_amplitude / target_amplitude.max()

    source_amplitude = np.ones_like(target_amplitude,dtype=np.float32)
    source_amplitude = source_amplitude / e

    inicio = time.perf_counter()
    recovered_phase, errors, int = gerchberg_saxton(source_amplitude, target_amplitude, epsilon_crit=1e-1, max_iter=50)
    final = time.perf_counter()
    print(f"Tiempo transcurrido: {(final - inicio)} segundos")


    with h5py.File('hpkGS.h5', 'w') as f:
        f.create_dataset('fase', data=recovered_phase)
        f.create_dataset('imagen', data=target_amplitude)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.title("Target Intensity")
    plt.imshow(target_amplitude**2, cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 4, 2)
    plt.title("Recovered Phase")
    plt.imshow(recovered_phase, vmin=-np.pi, vmax=np.pi)
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