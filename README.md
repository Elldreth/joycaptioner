# JoyCaptioner

**JoyCaptioner** is a PySide6 desktop application that leverages HuggingFace’s Llava model to generate long, formal, descriptive captions for images.

## Features

- **Folder Browser & History**: Load an entire folder of images and navigate via thumbnails in a docked history panel.
- **Dockable Panels**: Image history and captioner panels are dockable for flexible UI layouts.
- **Interactive Viewer**: Pan and zoom images smoothly with mouse controls or toolbar buttons.
- **Customizable Prompt System**: Choose from multiple prompt modes (Descriptive Caption, Straightforward, Stable Diffusion, MidJourney, Booru tag lists, Art Critic Analysis, Product Listing, Social Media Post, etc.) via a dropdown. Prompts auto-update based on mode, word count, length, and extra instructions.
- **Extra Prompt Customization**: Specify word count, caption length, and add extra instructions for highly tailored captions.
- **Single & Batch Captioning**: Generate a caption for the current image or batch-caption all images in the folder. Batch mode warns if caption files exist and asks for confirmation before overwriting.
- **Auto-Save & Auto-Load Captions**: Captions are saved as `.txt` files next to each image and are auto-loaded for editing when switching images.
- **Progress Dialog for Batch Processing**: Batch captioning runs in a separate thread and displays a progress dialog with a cancel button.
- **Resizable and Modern Window**: The app window opens at a modern standard size (1280x800) and cannot be resized below 1024x768.
- **Status Bar Feedback**: Status messages are shown for model loading, caption generation, saving, and errors.
- **Error Handling**: User-friendly error dialogs for missing models, failed captioning, or no images found.
- **About Dialog**: Accessible from the Help menu for app info and credits.
- **Modern Dark UI**: Fusion style with a dark palette and rounded controls for a sleek experience.

## Requirements

- Python 3.10 (recommended for your virtual environment)
- GPU with CUDA support (recommended for performance)
- `requirements.txt` (lists dependencies):
  ```
  torch
  transformers
  pillow
  nicegui>=1.3.14
  ```

## Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/Elldreth/joycaptioner.git
   cd joycaptioner
   ```
2. **Create & activate a virtual environment**
   - Windows:
     ```powershell
     python -m venv venv
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python joycaptioner.py
```

- Go to **File → Open Folder** to select an image directory.
- Click **Generate Caption** to create/edit a caption for the selected image.
- Use **Caption All Images in Folder** under **File** to batch-process.

## Configuration

- **Model Selection**: Change the `MODEL_NAME` constant in `joycaptioner.py` to switch to another Llava checkpoint.
- **Prompt Template**: Modify `DEFAULT_PROMPT` or enter a custom prompt in the UI before generation.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
