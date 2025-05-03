# Image to 3D Model Generation Script (Colab)

This script allows you to generate a basic 3D model (.obj, .stl) and a preview video from a 2D image using a specific implementation of the TripoSR model within a Google Colab environment.

## Steps to Run

1.  **Open Google Colab:** Go to [https://colab.research.google.com/](https://colab.research.google.com/) and open a new notebook.
2.  **Set Runtime:** Change the runtime type to include a GPU. Go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator.
3.  **Paste and Execute:** Copy the *entire* code block provided below into a single code cell in your Colab notebook.
4.  **Run the Cell:** Execute the cell.
5.  **Upload Image:** The script will pause and prompt you to upload an image file from your local computer. Select a `.png` or `.jpg` image of a single object.
6.  **Processing:** The script will then proceed with cloning the repository, installing dependencies, loading the model, processing your image, generating the 3D model, rendering a preview video, and extracting the mesh in .obj and .stl formats.
7.  **View Output:** Once execution is complete, the generated preview video (`render.mp4`) will be displayed directly in the notebook. The generated `.obj` and `.stl` files, along with render frames and the processed input image, will be saved in the `/content/TripoSR/output/0/` directory in the Colab file explorer. You can download these files from there.

## Libraries Used

The script installs and uses the following Python libraries:

* `torch`, `torchvision`, `torchaudio`: Core PyTorch library for model execution.
* `numpy`: For numerical operations, particularly image handling.
* `PIL (Pillow)`: For image loading and manipulation.
* `IPython.display.Video`: To display the output video in Colab.
* `rembg`: For background removal from the input image.
* `pymeshlab`: For mesh processing, specifically converting the generated .obj to .stl.
* `os`, `sys`, `time`, `shlex`, `subprocess`, `io`: Standard Python libraries for system interaction, timing, file operations, etc.
* `tsr`: The TripoSR library components cloned from the specified repository.

