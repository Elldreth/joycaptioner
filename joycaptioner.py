from nicegui import ui, app
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.image_utils import PILImageResampling # This might not be needed if using PIL.Image.Resampling directly
import os
import asyncio
import numpy as np
from typing import List, Dict, Optional, Callable, Any

# Handle Pillow resampling constant difference for compatibility
try:
    PIL_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    PIL_LANCZOS = Image.LANCZOS # For older Pillow versions

# ===== Constants and Configuration =====
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_PROMPT = "Write a long descriptive caption for this image in a formal tone."

# UI Configuration
UI_CONFIG = {
    "header_height": "4rem",
    "sidebar_width": "300px",
    "controls_width": "350px",
    "image_height": "calc(100vh - 12rem)",
    "caption_height": "200px"
}

# Prompt Templates
PROMPT_MODES = [
    "Descriptive Caption",
    "Straightforward Caption",
    "Stable Diffusion Prompt",
    "MidJourney",
    "Danbooru tag list",
    "e621 tag list",
    "Rule34 tag list",
    "Booru-Like Tag List",
    "Art Critic Analysis",
    "Product Listing",
    "Social Media Post",
]

PROMPT_TEMPLATES = {
    "Descriptive Caption": (
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
        "Write a detailed description for this image."
    ),
    "Straightforward Caption": (
        "Write a straightforward caption for this image in {word_count} words or less.",
        "Write a {length} straightforward caption for this image.",
        "Write a straightforward caption for this image."
    ),
    "Stable Diffusion Prompt": (
        "Write a stable diffusion prompt for this image in {word_count} words or less.",
        "Write a {length} stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image."
    ),
    "MidJourney": (
        "Write a MidJourney prompt for this image in {word_count} words or less.",
        "Write a {length} MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image."
    ),
    "Danbooru tag list": (
        "Write a comma-separated list of Danbooru tags for this image in {word_count} words or less.",
        "Write a {length} comma-separated list of Danbooru tags for this image.",
        "Write a comma-separated list of Danbooru tags for this image."
    ),
    "e621 tag list": (
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags."
    ),
    "Rule34 tag list": (
        "Write a comma-separated list of Rule34 tags for this image in {word_count} words or less.",
        "Write a {length} comma-separated list of Rule34 tags for this image.",
        "Write a comma-separated list of Rule34 tags for this image."
    ),
    "Booru-Like Tag List": (
        "Write a comma-separated list of booru-like tags for this image in {word_count} words or less.",
        "Write a {length} comma-separated list of booru-like tags for this image.",
        "Write a comma-separated list of booru-like tags for this image."
    ),
    "Art Critic Analysis": (
        "Write an art critic analysis for this image in {word_count} words or less.",
        "Write a {length} art critic analysis for this image.",
        "Write an art critic analysis for this image."
    ),
    "Product Listing": (
        "Write a product listing for this image in {word_count} words or less.",
        "Write a {length} product listing for this image.",
        "Write a product listing for this image."
    ),
    "Social Media Post": (
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post."
    ),
}

LENGTH_OPTIONS = ["any", "short", "medium", "long", "very long"]

# ===== Application Logic =====
class JoyCaptionerApp:
    """Main application logic for JoyCaptioner."""
    
    def __init__(self):
        self.processor = None
        self.llava_model = None
        self.image_paths: List[str] = []
        self.current_image_index: int = -1
        self.captions: Dict[str, str] = {}
        self.status: str = "Ready"
        self.load_model()

    def load_model(self) -> None:
        """Load the LLaVA model and processor"""
        try:
            self.status = "Loading model..."
            ui.notify(self.status) # type: ignore

            # Revert to default AutoProcessor to let it handle all image processing
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            
            # Determine model dtype: bfloat16 for CUDA, float32 for CPU
            if torch.cuda.is_available():
                model_dtype = torch.bfloat16 # Corrected to bfloat16 for CUDA
            else:
                model_dtype = torch.float32

            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=True
            )

            if torch.cuda.is_available():
                self.llava_model = self.llava_model.to('cuda')
                
            self.status = "Model loaded successfully"
            
        except Exception as e:
            self.status = f"Error loading model: {str(e)}"
            raise

    def set_image_folder(self, folder_path: str) -> bool:
        """Set the folder containing images to process"""
        try:
            if not os.path.isdir(folder_path):
                self.status = f"Folder not found: {folder_path}"
                return False
                
            self.image_paths = sorted([
                str(p) for p in Path(folder_path).glob('*')
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
            ])
            self.current_image_index = 0 if self.image_paths else -1
            self.status = f"Found {len(self.image_paths)} images" if self.image_paths else "No images found"
            return bool(self.image_paths)
        except Exception as e:
            self.status = f"Error loading folder: {str(e)}"
            return False

    def get_current_image_path(self) -> Optional[str]:
        """Get the path of the currently selected image"""
        if 0 <= self.current_image_index < len(self.image_paths):
            return self.image_paths[self.current_image_index]
        return None

    def generate_caption(self, image_path: str, prompt: str) -> str:
        """Generate a caption for the given image using the provided prompt"""
        try:
            if not os.path.exists(image_path):
                self.status = f"Image not found: {image_path}"
                return ""

            # Load and process image
            try:
                image = Image.open(image_path).convert('RGB')
                # Removed manual resize: image = image.resize((336, 336), resample=PIL_LANCZOS)
            except Exception as e:
                self.status = f"Error loading image: {str(e)}"
                return ""
                
            try:
                # Format the conversation with system and user messages
                convo = [
                    {"role": "system", "content": "You are a helpful image captioner."},
                    {"role": "user", "content": prompt},
                ]
                
                # Apply chat template
                convo_string = self.processor.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Process the inputs together - wrap text in a list
                inputs = self.processor(
                    text=[convo_string],  # Note: text is wrapped in a list
                    images=[image],        # Image is also wrapped in a list
                    return_tensors="pt"
                ).to('cuda')
                
                # Convert pixel values to bfloat16 if on CUDA
                if torch.cuda.is_available():
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16) # Corrected to bfloat16
                
                # Generate caption
                with torch.no_grad():
                    generate_ids = self.llava_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        suppress_tokens=None,
                        use_cache=True,
                        temperature=0.6,
                        top_k=None,
                        top_p=0.9,
                    )
                
                # Extract the generated text (skip the input part)
                generate_ids = generate_ids[0][inputs['input_ids'].shape[1]:]
                caption = self.processor.decode(generate_ids, skip_special_tokens=True)
                
                # Cache the caption
                self.captions[image_path] = caption
                self.status = "Caption generated successfully"
                return caption
                
            except Exception as e:
                self.status = f"Error processing image: {str(e)}"
                return ""
            
        except Exception as e:
            self.status = f"Error generating caption: {str(e)}"
            return ""

    def save_caption(self, image_path: str, caption: str) -> None:
        """Save caption to a file"""
        try:
            caption_file = Path(image_path).with_suffix('.txt')
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption)
        except Exception as e:
            self.status = f"Error saving caption: {str(e)}"

    def load_caption(self, image_path: str) -> str:
        """Load caption from file if it exists"""
        caption_file = Path(image_path).with_suffix('.txt')
        if caption_file.exists():
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except:
                return ""
        return ""

# ===== Global State =====
app_logic = None

# ===== UI Creation =====
# All UI-related functions are defined inside create_ui to ensure proper scoping

import asyncio

async def run_blocking(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

def create_ui():
    # Initialize global variables
    global app_logic
    
    # Initialize app logic
    app_logic = JoyCaptionerApp()
    
    # Define UI elements that need to be accessed by multiple functions
    folder_dialog = None
    image_list = None
    status_label = None
    image_display = None
    filename_label = None
    caption_output = None
    prompt_mode = None
    word_count = None
    caption_length = None
    extra_instructions = None
    about_dialog = None
    exit_dialog = None
    
    # ===== UI Event Handlers =====
    
    def show_folder_picker():
        """Show the folder selection dialog"""
        nonlocal folder_dialog
        folder_dialog.open()
    
    def on_folder_selected(folder_path: str, dialog):
        """Handle folder selection"""
        nonlocal status_label, image_list
        if not folder_path:
            status_label.text = 'Please enter a folder path'
            return
            
        if app_logic.set_image_folder(folder_path):
            update_image_list()
            if app_logic.image_paths:
                show_image(0)
            dialog.close()
        else:
            status_label.text = app_logic.status
    
    def update_image_list():
        """Update the image list in the sidebar"""
        nonlocal image_list
        image_list.clear()
        for idx, img_path in enumerate(app_logic.image_paths):
            with ui.row().classes('w-full items-center gap-2 p-2 hover:bg-gray-100 rounded cursor-pointer') as row:
                img_name = os.path.basename(img_path)
                ui.image(img_path).classes('w-12 h-12 object-cover rounded')
                ui.label(img_name).classes('truncate flex-1')
                row.on('click', lambda e, i=idx: show_image(i))
    
    def show_image(index: int):
        """Show the image at the given index"""
        nonlocal image_display, filename_label, caption_output, image_list
        if 0 <= index < len(app_logic.image_paths):
            app_logic.current_image_index = index
            img_path = app_logic.image_paths[index]
            image_display.set_source(img_path)
            filename_label.text = os.path.basename(img_path)
            
            # Load caption if exists
            caption = app_logic.load_caption(img_path)
            caption_output.value = caption if caption else ''
            
            # Update selected state in image list
            for i, child in enumerate(image_list):
                child.classes(replace='bg-blue-50' if i == index else '')
    
    def show_previous_image():
        """Show the previous image in the list"""
        if app_logic.current_image_index > 0:
            show_image(app_logic.current_image_index - 1)
    
    def show_next_image():
        """Show the next image in the list"""
        if app_logic.current_image_index < len(app_logic.image_paths) - 1:
            show_image(app_logic.current_image_index + 1)
    
    def generate_caption():
        """Generate a caption for the current image"""
        nonlocal status_label, caption_output
        if not app_logic.image_paths:
            status_label.text = 'No images loaded'
            return
            
        img_path = app_logic.get_current_image_path()
        if not img_path:
            status_label.text = 'No image selected'
            return
            
        prompt = build_prompt()
        status_label.text = 'Generating caption...'
        
        # Create a flag to track when generation is complete
        generation_complete = False
        
        def update_status():
            """Update the status label from the main thread"""
            nonlocal generation_complete
            status_label.text = app_logic.status
            if generation_complete:
                return False  # Stop the timer
            return True  # Continue the timer
        
        # Start a timer to update the status
        status_timer = ui.timer(0.1, update_status)
        
        async def generate_and_update():
            """Generate caption in background and update UI when done"""
            nonlocal generation_complete
            try:
                # Generate the caption (this runs in a background thread)
                caption = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: app_logic.generate_caption(img_path, prompt)
                )
                
                # Update UI in the main thread
                if caption:
                    caption_output.value = caption
                    app_logic.save_caption(img_path, caption)
                
                # Update status one final time
                status_label.text = app_logic.status
                
            except Exception as e:
                status_label.text = f'Error: {str(e)}'
                
            finally:
                # Stop the status update timer
                status_timer.deactivate()
                generation_complete = True
        
        # Start the background task
        asyncio.create_task(generate_and_update())
    
    def build_prompt() -> str:
        """Build the prompt based on UI selections"""
        nonlocal prompt_mode, word_count, caption_length, extra_instructions
        mode = prompt_mode.value
        word_count_val = int(word_count.value) if word_count.value else 50
        length = caption_length.value
        extra = extra_instructions.value.strip()
        
        template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["Descriptive Caption"])
        if length == "any":
            prompt = template[0].format(word_count=word_count_val, length=length)
        else:
            prompt = template[1].format(word_count=word_count_val, length=length)
            
        if extra:
            prompt += f"\n\nAdditional instructions: {extra}" # Newline for prompt string
            
        return prompt
    
    # Batch cancel flag and button reference
    batch_cancel_requested = False
    # Pre-create Cancel Batch button, hidden by default
    batch_cancel_button = ui.button('Cancel Batch', color='negative', on_click=lambda: cancel_batch()).props('hidden').classes('fixed bottom-16 right-6 z-50')

    # Utility to clear status bar and batch status label
    def clear_status():
        status_bar_label.set_text("")
        batch_status_label.set_text("")

    async def handle_generate_all_captions():
        nonlocal status_label, caption_output, image_display, filename_label, batch_cancel_requested, batch_cancel_button
        if not app_logic.image_paths:
            ui.notify("No images loaded to generate captions for.", type='warning')
            return
        
        prompt_text = build_prompt()
        total_images = len(app_logic.image_paths)
        images_processed = 0
        batch_cancel_requested = False

        status_label.set_text(f"Starting batch generation for {total_images} images...")
        status_bar_label.set_text("Batch captioning in progress... You may cancel at any time.")
        batch_status_label.set_text("")

        main_loop = asyncio.get_event_loop()

        # Show Cancel Batch button
        main_loop.call_soon_threadsafe(lambda: batch_cancel_button.props(remove='hidden'))

        def worker():
            nonlocal images_processed, batch_cancel_requested
            for idx, img_path in enumerate(app_logic.image_paths):
                if batch_cancel_requested:
                    break
                caption = app_logic.generate_caption(img_path, prompt_text)
                if caption:
                    app_logic.save_caption(img_path, caption)
                    images_processed += 1

                def update_ui_sync():
                    image_display.set_source(img_path)
                    filename_label.set_text(Path(img_path).name)
                    caption_output.set_value(caption if caption else "")
                    status_label.set_text(f"Processing {idx+1}/{total_images}: {Path(img_path).name}")
                main_loop.call_soon_threadsafe(update_ui_sync)

        await run_blocking(worker)

        # Hide Cancel Batch button
        main_loop.call_soon_threadsafe(lambda: batch_cancel_button.props('hidden'))

        def final_ui_update():
            if batch_cancel_requested:
                msg = f"Batch captioning canceled after {images_processed}/{total_images} images."
                status_label.set_text(msg)
                batch_status_label.set_text(msg)
                status_bar_label.set_text(msg)
            else:
                msg = f"Batch captioning complete: {images_processed}/{total_images} images processed."
                status_label.set_text(msg)
                batch_status_label.set_text(msg)
                status_bar_label.set_text(msg)
        main_loop.call_soon_threadsafe(final_ui_update)

    def cancel_batch():
        nonlocal batch_cancel_requested
        batch_cancel_requested = True
        status_bar_label.set_text("Cancel requested. Finishing current image...")
    
    def save_current_caption():
        """Save the current caption to file"""
        nonlocal status_label, caption_output
        if app_logic.current_image_index >= 0:
            img_path = app_logic.get_current_image_path()
            if img_path:
                app_logic.save_caption(img_path, caption_output.value)
                status_label.text = 'Caption saved'
    
    def show_about():
        """Show the about dialog"""
        nonlocal about_dialog
        about_dialog.open()
    
    def confirm_exit():
        """Show the exit confirmation dialog"""
        nonlocal exit_dialog
        exit_dialog.open()
    
    def exit_app():
        """Exit the application"""
        app.shutdown()
    
    async def run_async(func):
        """Run a function asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, func)

    # ===== Main UI Layout =====
    # Configure the app
    # Add CSS styles for the application
    ui.add_head_html('''
        <style>
            .sidebar { width: 300px; }
            .main-content { flex: 1; }
            .controls { width: 350px; }
            /* Removed .image-container and .image-container img rules as they might not apply here */
            .q-img__image { /* Target the actual class on the img tag generated by Quasar/NiceGUI */
                object-fit: contain !important;
                /* The w-full h-full classes on the ui.image component should handle width/height */
            }
            .image-thumbnail {
                width: 100%;
                height: 80px;
                object-fit: cover;
                border-radius: 4px;
                cursor: pointer;
            }
            .image-thumbnail:hover {
                opacity: 0.8;
            }
            .image-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
                min-height: 300px;
                background-color: #f9f9f9;
                border: 1px solid #eee;
                border-radius: 4px;
                overflow: hidden;
            }
        </style>
    ''')

    # Status bar at the bottom of the UI
    status_bar_label = ui.label('').classes('w-full text-center bg-gray-800 text-white py-2 fixed bottom-0 left-0 right-0 z-50')

    # Header
    with ui.header().classes('items-center justify-between bg-primary text-white p-4'):
        with ui.row().classes('items-center gap-4'):
            ui.icon('photo_camera', size='2rem')
            ui.label('JoyCaptioner').classes('text-2xl font-bold')
        
        with ui.row().classes('items-center gap-2'):
            ui.button('Open Folder', icon='folder', on_click=lambda: (clear_status(), show_folder_picker()))
            ui.button('About', icon='help', on_click=lambda: (clear_status(), show_about()))
            ui.button(icon='exit_to_app', on_click=lambda: (clear_status(), confirm_exit())).props('flat round')

    # Main content
    with ui.row().classes('w-full h-[calc(100vh-4rem)]'):
        # Sidebar
        with ui.column().classes('sidebar bg-gray-50 p-4 border-r border-gray-200 h-full overflow-y-auto'):
            ui.label('Image List').classes('text-lg font-semibold mb-2')
            image_list = ui.column().classes('gap-2')
            ui.space()
            ui.label('Status:').classes('text-sm font-medium mt-2')
            status_label = ui.label(app_logic.status).classes('text-sm text-gray-600')
            batch_status_label = ui.label('').classes('text-green-700 font-semibold mt-2')

        # Main content area
        with ui.column().classes('main-content p-4 h-full flex flex-col overflow-y-auto'): # ADDED flex flex-col
            with ui.card().classes('w-full flex-1 min-h-0 flex flex-col'): # CHANGED h-full to flex-1 min-h-0
                # This 'outer_column' holds the title, filename, image card, and nav buttons
                with ui.column().classes('flex-1 min-h-0 w-full flex flex-col items-center overflow-hidden'): # Added overflow-hidden
                    ui.label('Image Viewer').classes('text-lg font-semibold my-2 text-center shrink-0') # Title
                    filename_label = ui.label("No image selected").classes('text-center mb-2 shrink-0') # Filename
                    
                    # Image's card: directly grows within 'outer_column'
                    with ui.card().tight().classes('w-full flex-1 min-h-0 shadow-lg rounded-lg'): # REMOVED flex items-center justify-center
                        image_display = ui.image().classes('w-full h-full')
                    
                    # Navigation buttons
                    with ui.row().classes('w-full justify-center mt-2 shrink-0'): 
                        ui.button('Previous', icon='chevron_left', on_click=show_previous_image)
                        ui.button('Next', icon='chevron_right', on_click=show_next_image)

        # Controls sidebar
        with ui.column().classes('controls bg-gray-50 p-4 border-l border-gray-200 h-full overflow-y-auto'):
            ui.label('Caption Controls').classes('text-lg font-semibold mb-4')
            
            prompt_mode = ui.select(
                label='Prompt Mode',
                options=list(PROMPT_TEMPLATES.keys()),
                value='Descriptive Caption'
            ).classes('w-full mb-4')
            
            word_count = ui.number(
                label='Number of Words',
                value=55,
                min=1,
                max=500
            ).classes('w-full mb-4')
            
            caption_length = ui.select(
                label='Caption Length',
                options=LENGTH_OPTIONS,
                value='any'
            ).classes('w-full mb-4')
            
            extra_instructions = ui.textarea(
                label='Additional Instructions (optional)',
                placeholder='Enter any additional instructions here...'
            ).classes('w-full mb-4')
            
            generate_btn = ui.button(
                'Generate Caption',
                on_click=lambda: (clear_status(), generate_caption()),
                color='primary'
            ).classes('w-full mt-4')
            
            # Batch warning dialog setup
            batch_warning_dialog = ui.dialog()
            with batch_warning_dialog, ui.card().classes('w-full max-w-md p-4'):
                ui.label('Warning: Overwrite Captions').classes('text-xl font-bold text-amber-600')
                ui.label('Batch captioning will overwrite any existing captions in the folder.').classes('mb-2 text-amber-700')
                ui.label('Are you sure you want to proceed?').classes('mb-4')
                with ui.row().classes('w-full justify-end gap-2'):
                    ui.button('Cancel', on_click=batch_warning_dialog.close)
                    ui.button('Proceed', color='negative', on_click=lambda: (clear_status(), batch_warning_dialog.close(), asyncio.create_task(handle_generate_all_captions())))
            
            ui.button(
                'Generate All Captions',
                on_click=lambda: (clear_status(), batch_warning_dialog.open()),
                color='secondary'
            ).classes('w-full mt-2')
            
            ui.separator().classes('my-4')
            
            ui.label('Caption:').classes('font-medium')
            caption_output = ui.textarea(
                label='Generated Caption',
                placeholder='Caption will appear here...',
                on_change=save_current_caption
            ).props('filled autogrow').classes('w-full mt-2')

    # Dialogs
    folder_dialog = ui.dialog()
    with folder_dialog, ui.card().classes('w-full max-w-md p-4'):
        ui.label('Select Image Folder').classes('text-lg font-semibold mb-4')
        with ui.column().classes('w-full gap-4'):
            ui.label('Please select the folder containing your images:')
            folder_input = ui.input(
                label='Folder Path',
                placeholder='Paste the full folder path here...'
            ).classes('w-full')
            
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=folder_dialog.close)
                ui.button(
                    'Open',
                    on_click=lambda: on_folder_selected(folder_input.value, folder_dialog),
                    color='primary'
                )

    about_dialog = ui.dialog()
    with about_dialog, ui.card().classes('w-full max-w-md p-4'):
        with ui.column().classes('w-full gap-4'):
            ui.label('About JoyCaptioner').classes('text-xl font-bold')
            ui.label('Version 1.0.0')
            ui.label('A tool for generating captions for your images using AI')
            ui.label(' 2023 JoyCaptioner').classes('text-gray-500')
            ui.button('Close', on_click=about_dialog.close, color='primary').props('flat')

    exit_dialog = ui.dialog()
    with exit_dialog, ui.card().classes('w-full max-w-md p-4'):
        with ui.column().classes('w-full gap-4'):
            ui.label('Exit JoyCaptioner').classes('text-xl font-bold')
            ui.label('Are you sure you want to exit?')
            ui.label('Any unsaved changes will be lost.').classes('text-amber-600')
            
            with ui.row().classes('w-full justify-end gap-2'):
                ui.button('Cancel', on_click=exit_dialog.close)
                ui.button('Exit', on_click=exit_app, color='negative')

    # Initialize reactive variables for state management
    current_image = ''
    current_caption = ''

    # Event handlers
    def show_folder_picker():
        """Show the folder selection dialog"""
        folder_dialog.open()

    def on_folder_selected(folder_path: str, dialog):
        """Handle folder selection"""
        nonlocal status_label # image_list is already nonlocal due to update_image_list
        if not folder_path:
            ui.notify('Please enter a folder path', type='warning')
            # status_label.text = 'Please enter a folder path' # Covered by notify
            return
            
        if app_logic.set_image_folder(folder_path):
            if app_logic.image_paths:
                select_image_by_index_ui(0) # This will also call update_image_list_ui
            else:
                # Handle case where folder is valid but no images found
                app_logic.current_image_index = -1
                _display_current_image_ui() # Clear display
                update_image_list_ui()    # Update list to show "No images found"
                ui.notify('No supported images found in the selected folder.', type='info')
            dialog.close()
        else:
            ui.notify(app_logic.status, type='negative')
            # status_label.text = app_logic.status # Covered by notify

    def update_image_list_ui():
        nonlocal image_list # app_logic is global
        image_list.clear()
        if not app_logic.image_paths:
            with image_list:
                ui.label("No images found.").classes('text-gray-500 p-2')
            return

        for i, image_path_str in enumerate(app_logic.image_paths):
            image_path_obj = Path(image_path_str)
            with image_list:
                is_selected = (i == app_logic.current_image_index)
                
                base_row_classes = 'w-full p-1 cursor-pointer hover:bg-gray-100 rounded-md flex items-center gap-2 transition-colors duration-150 ease-in-out'
                selected_row_classes = 'bg-blue-100 border-l-4 border-blue-500 shadow-sm'
                
                row_classes = base_row_classes
                if is_selected:
                    row_classes += f' {selected_row_classes}'
                else:
                    row_classes += ' border-l-4 border-transparent' # Keep space for border consistency

                with ui.row().classes(row_classes).on('click', lambda e, captured_idx=i: select_image_by_index_ui(captured_idx)):
                    ui.image(image_path_str).classes('w-16 h-16 object-cover rounded-sm flex-shrink-0')
                    # ui.label(image_path_obj.name).classes('text-xs truncate flex-grow min-w-0') # Removed filename label

    def _display_current_image_ui():
        nonlocal image_display, filename_label, status_label, caption_output # Added caption_output
        if 0 <= app_logic.current_image_index < len(app_logic.image_paths):
            current_image_path = app_logic.image_paths[app_logic.current_image_index]
            image_display.set_source(current_image_path)
            filename_label.set_text(Path(current_image_path).name)
            app_logic.status = f"Viewing: {Path(current_image_path).name}"

            # Load and display caption
            caption = app_logic.load_caption(current_image_path)
            caption_output.set_value(caption if caption else '')
        else:
            image_display.set_source('') 
            filename_label.set_text("No image selected")
            caption_output.set_value('') # Clear caption output
            app_logic.status = "No image selected or folder empty"
        status_label.set_text(app_logic.status)

    def select_image_by_index_ui(index: int):
        nonlocal status_label # app_logic is global
        if not app_logic.image_paths:
            app_logic.current_image_index = -1
            _display_current_image_ui()
            update_image_list_ui()
            return

        if not (0 <= index < len(app_logic.image_paths)):
            if app_logic.image_paths:
                 app_logic.current_image_index = 0
            else:
                 app_logic.current_image_index = -1
        else:
            app_logic.current_image_index = index
        
        _display_current_image_ui()
        update_image_list_ui() 

    # --- Generate Caption and related ---
    def build_prompt() -> str:
        """Build the prompt based on UI selections"""
        nonlocal prompt_mode, word_count, caption_length, extra_instructions
        mode = prompt_mode.value
        word_count_val = int(word_count.value) if word_count.value else 50
        length = caption_length.value
        extra = extra_instructions.value.strip()
        
        template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["Descriptive Caption"])
        if length == "any":
            prompt = template[0].format(word_count=word_count_val, length=length)
        else:
            prompt = template[1].format(word_count=word_count_val, length=length)
            
        if extra:
            prompt += f"\n\nAdditional instructions: {extra}" # Newline for prompt string
            
        return prompt

    def generate_caption():
        """Generate a caption for the current image"""
        nonlocal status_label, caption_output, prompt_mode, word_count, caption_length, extra_instructions
        if not app_logic.image_paths:
            status_label.set_text('No images loaded')
            return
            
        img_path = app_logic.get_current_image_path()
        if not img_path:
            status_label.set_text('No image selected')
            return
            
        prompt_text = build_prompt()
        status_label.set_text('Generating caption...') 
        main_loop = asyncio.get_event_loop()

        def generate_and_update_sync(): 
            caption_result = app_logic.generate_caption(img_path, prompt_text)
            async def update_ui_after_generation():
                if caption_result:
                    caption_output.set_value(caption_result)
                    app_logic.save_caption(img_path, caption_result) 
                status_label.set_text(app_logic.status) 
            main_loop.call_soon_threadsafe(lambda: asyncio.create_task(update_ui_after_generation()))
        
        asyncio.create_task(run_blocking(generate_and_update_sync)) 

    def save_current_caption():
        """Save the current caption to file"""
        nonlocal status_label, caption_output 
        if app_logic.current_image_index >= 0:
            img_path = app_logic.get_current_image_path()
            if img_path:
                app_logic.save_caption(img_path, caption_output.value)
                status_label.set_text('Caption saved')

    # --- Dialogs and App Control ---
    def show_about():
        """Show the about dialog"""
        nonlocal about_dialog 
        about_dialog.open()

    def confirm_exit():
        """Show the exit confirmation dialog"""
        nonlocal exit_dialog 
        exit_dialog.open()

    def exit_app():
        """Exit the application"""
        # nicegui.app.shutdown() is the correct way to call this
        app.shutdown()

    # Initial UI update for status
    status_label.set_text(app_logic.status)
    
    # Create and run the UI
    ui.run(title="JoyCaptioner", port=8080, reload=False)

# Create and run the UI
if __name__ in {"__main__", "__mp_main__"}:
    create_ui()