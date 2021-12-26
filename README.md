# Creating Hybrid Images - Python Code #
At the very first step, an image filtering function has been developed from scratch accepting an image and a Gaussian filter kernel as its inputs. The second step was to write another function (from scratch again) to generate hybrid images. This function uses the developed image filtering function accepting two relevant images and the Gaussian filter kernel as its inputs. Also, given that a Gaussian filter is provided for us,  but as an interesting part of this repoitory, a faction has been developed to create a Gaussian filter kernel from scratch. This way one is able to simply understand the whole idea behind the provided filter.

### Image Filtering ###
Why we filter images? In this part, I try to answer this question. An image is nothing but a 2D signal and can be modeled mathematically by the f(x, y) function. x and y are the exact locations on each pixel in the image. The output of this function is an uint8 integer value from 0 to 255. So basically, an image is simply a matrix and each element of this matrix has the pixel value stored in it. Having said that there exist 2 image types: one is grayscale and the other is colored. A grayscale image is a 2D matrix and a colored image is a 3D matrix and each dimension of this 3D matrix is one color channel of red, green, and blue. That is why we call it the RGB image. My algorithm will support both grayscale and RGB images as input.

As we discussed, the image is a signal and each signal carries some information in form of frequency. So basically, image processing is a specific sort of Signal Processing. Specifically, dealing with images, we are filtering for the following reasons:
* Reducing the noise
* feature detection (Edge detection specifically)
* Blurring (Low-pass signal filtering)
* Sharpening (High-pass signal filtering)

In this repository, the goal is creating a low-pass filtering algorithm to make an input image clear from its high-frequency signals. The generated low-frequency image will be used to generate the high-frequency image. So that is gonna be incredibly easy. Here are the steps:
* Create a low-pass Gaussian filter
* Use the created filter to generate a blurred(e.g., low-frequency) image
* Use the blurred image to generate a sharpened (e.g., high-frequency) image by subtracting it from its original image
### How do I get set up? ###

* Summary of set up
* Configuration
* Dependencies
* Database configuration
* How to run tests
* Deployment instructions

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Behnam Moradi (behnammoradi026@gmail.com)
