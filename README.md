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

## Your Script Code

```python
%cd /content
# IMPORTANT: Replace the URL below with the actual GitHub repository URL you are using
# Example: !git clone [https://github.com/VAST-AI-Research/TripoSR.git](https://github.com/VAST-AI-Research/TripoSR.git)
!git clone [https://github.com/pyimagesearch/TripoSR.git](https://github.com/pyimagesearch/TripoSR.git) # Placeholder - replace with actual URL
%cd TripoSR

import sys
# Add the 'tsr' package directory inside the cloned repo to the Python path
sys.path.append('/content/TripoSR/tsr')
print(f"Added /content/TripoSR/tsr to sys.path: {sys.path}")

# Install requirements from the repository's requirements.txt
print("Installing requirements from requirements.txt...")
!pip install -r requirements.txt

# Install additional required libraries
print("Installing additional libraries...")
!pip install torch torchvision torchaudio # Core PyTorch libraries
# !pip install tsr # <-- This line is likely incorrect and installs a different package, usually not needed if cloning the repo
!pip install rembg[gpu] # Install rembg with GPU support if using a GPU runtime
!pip install pymeshlab
print("Additional libraries installed.")

# Change directory again to the 'tsr' package folder for direct imports like 'from system import TSR'
# Note: Having /content/TripoSR/tsr in sys.path should be enough, but this command is in the original script
%cd /content/TripoSR/tsr

# Import TSR and utilities from the cloned 'tsr' package
from system import TSR
from utils import remove_background, resize_foreground, save_video


# Timer class definition (as provided in your script)
class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0 # ms
        self.time_unit = "ms"
    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        print(f"{name} finished in {t:.2f}{self.time_unit}.")
timer = Timer()

# --- Image Upload and Preprocessing ---
from google.colab import files
uploaded = files.upload()
original_image = Image.open(list(uploaded.keys())[0])

# Save uploaded image to a predefined location within the repo structure
# Note: This assumes the 'examples' directory exists in the cloned repo root
image_save_path = "/content/TripoSR/examples/product.png"
os.makedirs(os.path.dirname(image_save_path), exist_ok=True) # Ensure examples dir exists
original_image.resize((512, 512)).save(image_save_path)
print(f"Uploaded image saved to {image_save_path}")


# --- Configuration ---
# These variables control the model, processing, and output
image_paths = image_save_path # Use the path where the uploaded image was saved
device = "cuda:0" if torch.cuda.is_available() else "cpu" # Ensure device is correctly set based on runtime
pretrained_model_name_or_path = "stabilityai/TripoSR" # Model weights source (Hugging Face ID)
chunk_size = 8192 # Rendering parameter
no_remove_bg = True # Flag in script, though remove_background is still called
foreground_ratio = 0.85 # Preprocessing parameter
output_dir = "output/" # Output directory relative to current location (/content/TripoSR/tsr/)
model_save_format = "obj"
render = True # Whether to render video/frames
output_dir_abs = os.path.join("/content/TripoSR", output_dir.strip()) # Absolute path for output
os.makedirs(output_dir_abs, exist_ok=True)
print(f"Output will be saved to: {output_dir_abs}")


# --- Model Initialization ---
timer.start("Initializing model")
# Load the model from Hugging Face using from_pretrained
model = TSR.from_pretrained(
    pretrained_model_name_or_path,
    config_name="config.yaml", # Expected config file name in the HF repo
    weight_name="model.ckpt", # Expected weight file name in the HF repo
)
model.renderer.set_chunk_size(chunk_size)
model.to(device)
timer.end("Initializing model")

# --- Image Processing (using utility functions from the repo) ---
timer.start("Processing images")
images = []
rembg_session = rembg.new_session() # Initialize rembg session

# Call utility functions from the cloned repo
# Note: The script calls remove_background and resize_foreground sequentially on original_image
image = remove_background(original_image, rembg_session)
image = resize_foreground(original_image, foreground_ratio)

# Handle potential RGBA output - convert to RGB with gray background (0.5)
if image.mode == "RGBA":
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5 # Alpha composite with gray
    image = Image.fromarray((image_np * 255.0).astype(np.uint8))

# Save the processed input image
image_output_subdir = os.path.join(output_dir_abs, str(0))
os.makedirs(image_output_subdir, exist_ok=True)
processed_input_path = os.path.join(image_output_subdir, "input.png")
image.save(processed_input_path)
images.append(image) # Add the processed image to the list


timer.end("Processing images")

# --- 3D Generation, Rendering, and Mesh Extraction ---
for i, image in enumerate(images): # Loops through the processed image list (contains 1 image)
    print(f"Running image {i + 1}/{len(images)} ...")

    # Run the core model to get scene codes (latent 3D representation)
    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device) # Model takes a list of images
    timer.end("Running model")

    # Render preview video and frames if enabled
    if render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil") # Render 30 views
        render_output_dir = os.path.join(output_dir_abs, str(i))
        os.makedirs(render_output_dir, exist_ok=True)
        for ri, render_image in enumerate(render_images[0]): # Save individual frames
             render_image.save(os.path.join(render_output_dir, f"render_{ri:03d}.png"))
        save_video( # Save video using utility function
            render_images[0], os.path.join(render_output_dir, "render.mp4"), fps=30
        )
        timer.end("Rendering")

    # Extract the 3D mesh from the scene codes
    timer.start("Exporting mesh")
    meshes = model.extract_mesh(scene_codes, has_vertex_color=False) # Extracts list of meshes
    mesh_file_base = os.path.join(output_output_subdir, f"mesh.{model_save_format}")
    meshes[0].export(mesh_file_base) # Export the first mesh
    timer.end("Exporting mesh")

print("Processing complete.")

# --- Convert OBJ to STL using PyMeshLab ---
# Note: This step uses PyMeshLab to load the generated OBJ and save it as STL
obj_file_path = os.path.join(output_output_subdir, f"mesh.{model_save_format}") # Path to the saved OBJ
stl_file_path = os.path.join(output_output_subdir, f"mesh.stl") # Path for the output STL

try:
    print(f"\nConverting {obj_file_path} to STL...")
    ms = pymesh.MeshSet()
    ms.load_new_mesh(obj_file_path)
    # mesh = ms.current_mesh() # Get the current mesh if needed for other operations
    ms.save_current_mesh(stl_file_path) # Save in STL format
    print(f"Converted mesh saved as {stl_file_path}")
except Exception as e:
    print(f"Error during OBJ to STL conversion: {e}")
    print("Please ensure the OBJ file was generated correctly and PyMeshLab is installed.")


# --- Display Output Video ---
# Display the generated render video in the Colab notebook
print("\nDisplaying render video:")
display(Video(os.path.join(output_output_subdir, "render.mp4"), embed=True))

# --- Instructions for Downloading Output ---
print(f"\nGenerated files (.obj, .stl, .mp4, .png) are saved in: {output_output_subdir}")
print("You can download these files using the file browser on the left sidebar in Google Colab.")
