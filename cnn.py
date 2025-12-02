import numpy as np

# Toy Dataset: Random 8x8 image
image = np.random.randn(8, 8)

class CNN_Layer:
    def __init__(self, kernel_size, num_filters):
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.kernels = np.random.randn(num_filters, kernel_size, kernel_size) / (kernel_size**2)

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.kernel_size + 1):
            for j in range(w - self.kernel_size + 1):
                im_region = image[i:(i + self.kernel_size), j:(j + self.kernel_size)]
                yield im_region, i, j

    def forward(self, input):
        h, w = input.shape
        output = np.zeros((h - self.kernel_size + 1, w - self.kernel_size + 1, self.num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.kernels, axis=(1, 2))
            
        return output

# Execution
conv = CNN_Layer(kernel_size=3, num_filters=1)
output = conv.forward(image)
print("\nCNN Output Shape:", output.shape)
print("CNN Feature Map (Subset):\n", output[:3, :3, 0])