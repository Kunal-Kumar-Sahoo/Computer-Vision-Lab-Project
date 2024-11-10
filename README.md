# Computer Vision Lab Demo Application

This is a Streamlit application that provides various computer vision functionalities, including edge detection, image binarization, image denoising, histogram equalization, histogram matching, and image domain transformations (DFT and DCT). The application allows users to upload images and apply different processing techniques interactively.

## Features

- **Edge Detection**: Supports Canny, Sobel, and Roberts methods.
- **Image Binarization**: Offers manual thresholding, adaptive mean thresholding, and adaptive Gaussian thresholding.
- **Image Denoising**: Implements mean filter, median filter, and Gaussian filter.
- **Histogram Equalization**: Enhances the contrast of images.
- **Histogram Matching**: Matches the histogram of one image to another.
- **Image Domain Transformation**: Performs Discrete Fourier Transform (DFT) and Discrete Cosine Transform (DCT).

## Requirements

To run this application locally or in a Docker container, you need to have the following software installed:

- Python 3.7 or higher
- pip (Python package installer)
- Docker (if using Docker)

## Installation

### Local Installation

1. Clone the repository or download the source code.

   ```bash
   git clone https://github.com/Kunal-Kumar-Sahoo/Computer-Vision-Lab-Project.git
   cd Computer-Vision-Lab-Project/
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application Locally

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to [http://localhost:8501](http://localhost:8501) to access the application.

## Docker Setup

If you prefer to run the application in a Docker container, follow these steps:

### Building the Docker Image

1. Ensure you are in the project directory where the `Dockerfile` is located.

2. Build the Docker image:

   ```bash
   docker build -t streamlit-app .
   ```

### Running the Docker Container

1. Run the Docker container:

   ```bash
   docker run -p 8501:8501 streamlit-app
   ```

2. Open your web browser and go to [http://localhost:8501](http://localhost:8501) to view the application.


## Usage

Once you have launched the application, you can upload an image using the file uploader provided in the interface. Select from various processing options available in each section to apply transformations or analyses on your uploaded image.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.