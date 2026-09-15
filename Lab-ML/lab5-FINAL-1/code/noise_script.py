from PIL import Image
import numpy as np

# Open the image and ensure that it is greyscale.
image = Image.open("greyscale.jpg").convert("L")

# Convert the image to an array.
# int16 allows safe addition of negative noise.
I = np.array(image, dtype=np.int16)

# Generate independent uniform integer noise in [-15, 15].
N = np.random.randint(-15, 16, size=I.shape)

# Add noise to the original image.
O = I + N

# Clip values outside the valid greyscale interval.
O = np.clip(O, 0, 255)

# Convert the result back to an 8-bit greyscale image.
noisy_image = Image.fromarray(O.astype(np.uint8), mode="L")

# Save the noisy image.
noisy_image.save("greyscale_noisy.jpg")

print("Noisy image saved as greyscale_noisy.jpg")