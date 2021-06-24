import numpy as np
import matplotlib.pylab as plt
import cv2

# Function: Used to read images
def import_imgs():
  print("Importing Images...")
  img_names = [['submarine.bmp', 'fish.bmp'], ['plane.bmp', 'bird.bmp'], ['cat.bmp', 'dog.bmp'], ['einstein.bmp', 'marilyn.bmp'], ['bicycle.bmp', 'motorcycle.bmp']]
  imgs_list = []
  for pair in img_names:
    for img_name in pair:
      img_path = "../data/" + img_name
      imgs_list.append(plt.imread(img_path))
  return imgs_list

# Function: Will receive a list of image and it will show all of them
def show_img(img):
  print("Ploting Images...")
  number_of_images = len(img)
  n = 0
  for i in img:
    plt.subplot(1, number_of_images, n+1) 
    plt.imshow(i)
    n = n + 1
  plt.show()

# Function: Will create gaussian kernel
def create_gaussian_filter(kernel_size=3, sigma=1.0):
  kernel = np.zeros((kernel_size,kernel_size))
  i = -int(kernel_size/2)
  for r in range(0, kernel_size):
    j = -int(kernel_size/2)
    for c in range(0, kernel_size):
      kernel[r, c] = (1.0/(2.0*np.pi*sigma*sigma))*np.exp(-(i**2 + j**2)/(2.0*sigma*sigma))
      j = j+1
    i = i+1
  # Kernel Normalization
  kernel = (kernel)/(kernel.max())
  return kernel

# Function: Will create image padding
def img_padding(img_data, kernel):
  padding_number = int(int(np.sqrt((kernel.size)))/2)
  print("Padding Number = " + str(padding_number))
  image_shape = img_data.shape
  if len(image_shape) == 3:
    i, j, k = img_data.shape
    zeros_column = np.zeros((1, i))
    zeros_row =  np.zeros((1, j + 2*padding_number))

    red_channel = img_data[:,:,0]
    green_channel = img_data[:,:,1]
    blue_channel = img_data[:,:,2]

    # for loop: to add zero padding to columns
    counter = 1
    for l in range(0, padding_number):

      red_channel = np.insert(red_channel, 0, zeros_column, axis=1)
      red_channel = np.insert(red_channel, j + counter , zeros_column, axis=1)

      blue_channel = np.insert(blue_channel, 0, zeros_column, axis=1)
      blue_channel = np.insert(blue_channel, j + counter , zeros_column, axis=1)

      green_channel = np.insert(green_channel, 0, zeros_column, axis=1)
      green_channel = np.insert(green_channel, j + counter , zeros_column, axis=1)

      counter = counter + 1

    # for loop: to add zero padding to rows
    counter = 1
    for p in range(0, padding_number):
      red_channel = np.insert(red_channel, 0, zeros_row, axis=0)
      red_channel = np.insert(red_channel, i + counter , zeros_row, axis=0)

      blue_channel = np.insert(blue_channel, 0, zeros_row, axis=0)
      blue_channel = np.insert(blue_channel, i + counter , zeros_row, axis=0)

      green_channel = np.insert(green_channel, 0, zeros_row, axis=0)
      green_channel = np.insert(green_channel, i + counter , zeros_row, axis=0)

      counter = counter + 1
    paded_image = np.stack((red_channel, green_channel, blue_channel), axis=2)
    return paded_image

  # if condition: grayscale images
  if len(image_shape) != 3:
    i, j= img_data.shape
    zeros_column = np.zeros((1, i))
    zeros_row =  np.zeros((1, j + 2*padding_number))
    # for loop: to add zero padding to columns
    counter = 1
    for l in range(0, padding_number):

      img_data = np.insert(img_data, 0, zeros_column, axis=1)
      img_data = np.insert(img_data, j + counter , zeros_column, axis=1)
      counter = counter + 1

    # for loop: to add zero padding to rows
    counter = 1
    for p in range(0, padding_number):

      img_data = np.insert(img_data, 0, zeros_row, axis=0)
      img_data = np.insert(img_data, i + counter , zeros_row, axis=0)
      counter = counter + 1
    return img_data

# Function: will filter the inpute image. It supports colored and grayscale images
def my_imfilter(image, filter):
  image = img_padding(image, filter)
  # show_img([])
  # filter = np.dot(filter, filter.T)
  pad_number = int(int(np.sqrt((filter.size)))/2)
  image_shape = image.shape
  if len(image_shape) == 3:
    i, j, k = image.shape
    low_pass_filtered_img = image.copy()
    for dim in range(0, k):
      print("dim = " + str(dim))
      for row in range(0 + pad_number, i - pad_number):
        for col in range(0 + pad_number, j - pad_number):
          sep_data = image[row - pad_number:row + pad_number+1, col - pad_number:col + pad_number+1, dim]*filter
          low_pass_filtered_img[row, col, dim] = np.sum(sep_data)/filter.size
    return low_pass_filtered_img[0 + pad_number:i - pad_number, 0 + pad_number:j - pad_number]

  if len(image_shape) != 3:
    i, j = image.shape
    low_pass_filtered_img = image.copy()
    for row in range(0 + pad_number, i - pad_number):
      for col in range(0 + pad_number, j - pad_number):
        sep_data = image[row - pad_number:row + pad_number+1, col - pad_number:col + pad_number+1]*filter
        low_pass_filtered_img[row, col] = np.sum(sep_data)/filter.size
    return low_pass_filtered_img[0 + pad_number:i - pad_number, 0 + pad_number:j - pad_number]

# Function: Will create hybrid images. It supports colored and grayscale images
def create_hybrid_image(image1, image2, filter):
  # hybrid_image = image1 + image2
  low_freq = my_imfilter(image1, filter)
  high_freq = image2 - my_imfilter(image2, filter)
  hybrid_image = high_freq + low_freq
  return[low_freq, high_freq, hybrid_image]
