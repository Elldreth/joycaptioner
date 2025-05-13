# JoyCaptioner

**JoyCaptioner** is a PySide6 desktop application that leverages HuggingFace’s Llava model to generate long, formal, descriptive captions for images.

## Features

- **Folder Browser & History**: Load an entire folder of images and navigate via thumbnails in a docked history panel.
- **Interactive Viewer**: Pan and zoom images smoothly with mouse controls or toolbar buttons.
- **Customizable Prompt**: Enter your own caption prompt or use the default formal-tone template.
- **Single & Batch Captioning**: Generate a caption for the current image or batch-caption all images in the folder (with overwrite protection).
- **Auto-Save Captions**: Captions are saved automatically as `.txt` files alongside each image.
- **Modern Dark UI**: Fusion style with a dark palette and rounded controls for a sleek experience.

## Requirements

- Python 3.10 (recommended for your virtual environment)
- GPU with CUDA support (recommended for performance)
- `requirements.txt` (lists dependencies):
  ```
  torch
  transformers
  pillow
  PySide6
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
