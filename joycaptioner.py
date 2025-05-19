import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap, QAction, QIcon, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout,
    QHBoxLayout, QFileDialog, QLineEdit, QSizePolicy, QMessageBox, QListWidget, QListWidgetItem,
    QDockWidget, QToolBar, QStatusBar, QMenuBar, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QDialog, QProgressBar, QDialogButtonBox
)
from PySide6.QtCore import QThread, Signal, QObject

MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"
DEFAULT_PROMPT = "Write a long descriptive caption for this image in a formal tone."

class ImageHistoryDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Image History", parent)
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setIconSize(QSize(96, 96))
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setMovement(QListWidget.Static)
        self.list_widget.setSpacing(8)
        self.setWidget(self.list_widget)
        self.setAllowedAreas(Qt.LeftDockWidgetArea)

    def add_image(self, image_path):
        # Show thumbnail as icon, store image_path as data
        from PySide6.QtGui import QPixmap, QIcon
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            icon = QIcon(pixmap.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            icon = QIcon()
        item = QListWidgetItem(icon, "")
        item.setData(Qt.UserRole, image_path)
        self.list_widget.addItem(item)

    def clear_history(self):
        self.list_widget.clear()

class CaptionerDock(QDockWidget):
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
        # Each entry is a tuple: (with word_count, with length, default)
        "Descriptive Caption": (
            "Write a detailed description for this image in {word_count} words or less.",
            "Write a {length} detailed description for this image.",
            "Write a long detailed description for this image."
        ),
        "Straightforward Caption": (
            "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
            "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
            "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing."
        ),
        "Stable Diffusion Prompt": (
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
            "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt."
        ),
        "MidJourney": (
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image."
        ),
        "Danbooru tag list": (
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: artist:, copyright:, character:, meta:, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text."
        ),
        "e621 tag list": (
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags."
        ),
        "Rule34 tag list": (
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags."
        ),
        "Booru-Like Tag List": (
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image."
        ),
        "Art Critic Analysis": (
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc."
        ),
        "Product Listing": (
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing."
        ),
        "Social Media Post": (
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post."
        ),
    }
    LENGTH_OPTIONS = ["any", "short", "medium", "long", "very long"]

    def __init__(self, parent=None, default_prompt=DEFAULT_PROMPT):
        super().__init__("Captioner", parent)
        from PySide6.QtWidgets import QComboBox, QHBoxLayout
        self.panel = QWidget()
        layout = QVBoxLayout()

        # Prompt mode dropdown
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Prompt Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.PROMPT_MODES)
        self.mode_combo.setMinimumWidth(280)  # Make dropdown wide enough for all options
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Word count field
        word_layout = QHBoxLayout()
        word_label = QLabel("Number of Words:")
        self.word_input = QLineEdit()
        self.word_input.setText("55")
        self.word_input.setMaximumWidth(60)
        word_layout.addWidget(word_label)
        word_layout.addWidget(self.word_input)
        layout.addLayout(word_layout)

        # Caption length dropdown
        length_layout = QHBoxLayout()
        length_label = QLabel("Caption Length:")
        self.length_combo = QComboBox()
        self.length_combo.addItems(self.LENGTH_OPTIONS)
        length_layout.addWidget(length_label)
        length_layout.addWidget(self.length_combo)
        layout.addLayout(length_layout)

        # Additional instructions field
        self.extra_instructions_edit = QTextEdit()
        self.extra_instructions_edit.setPlaceholderText("Additional instructions (optional)...")
        self.extra_instructions_edit.setMaximumHeight(50)
        layout.addWidget(QLabel("Additional Instructions:"))
        layout.addWidget(self.extra_instructions_edit)

        # Hidden prompt input (not shown to user)
        self.prompt_input = QLineEdit(default_prompt)
        self.prompt_input.setVisible(False)
        layout.addWidget(self.prompt_input)

        self.caption_btn = QPushButton("Generate Caption")
        layout.addWidget(self.caption_btn)
        self.caption_edit = QTextEdit()
        self.caption_edit.setReadOnly(True)
        self.caption_edit.setPlaceholderText("Caption will appear here...")
        layout.addWidget(QLabel("Caption:"))
        layout.addWidget(self.caption_edit)
        self.panel.setLayout(layout)
        self.setWidget(self.panel)
        self.setAllowedAreas(Qt.RightDockWidgetArea)

        # Connect changes to update the prompt
        self.mode_combo.currentIndexChanged.connect(self.update_prompt)
        self.word_input.textChanged.connect(self.update_prompt)
        self.length_combo.currentIndexChanged.connect(self.update_prompt)
        self.extra_instructions_edit.textChanged.connect(self.update_prompt)
        self.update_prompt()

    def update_prompt(self):
        mode = self.mode_combo.currentText()
        word_count = self.word_input.text().strip()
        length = self.length_combo.currentText()
        extra = self.extra_instructions_edit.toPlainText().strip()
        template_wc, template_len, template_default = self.PROMPT_TEMPLATES.get(mode, (DEFAULT_PROMPT, DEFAULT_PROMPT, DEFAULT_PROMPT))
        prompt = template_default
        if length != "any":
            prompt = template_len.replace("{length}", length)
        elif word_count.isdigit() and int(word_count) > 0:
            prompt = template_wc.replace("{word_count}", word_count)
        else:
            prompt = template_default
        if extra:
            prompt = prompt.rstrip() + " " + extra.lstrip()
        self.prompt_input.setText(prompt)


class JoyCaptionMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JoyCaptioner - Image Captioner")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 800)
        self.processor = None
        self.llava_model = None
        self.image = None
        self.image_path = None
        self._prev_image_path = None
        self._prev_caption = None
        self.history = []
        self.init_ui()
        self.load_model()

    def init_ui(self):
        # Central Image Viewer with filename label below
        self.graphics_view = ImageGraphicsView()
        self.filename_label = QLabel("")
        self.filename_label.setAlignment(Qt.AlignCenter)
        self.filename_label.setStyleSheet("font-size: 16px; margin-top: 10px;")
        self.viewer_panel = QWidget()
        viewer_layout = QVBoxLayout()
        viewer_layout.addWidget(self.graphics_view, stretch=1)
        viewer_layout.addWidget(self.filename_label, stretch=0)
        self.viewer_panel.setLayout(viewer_layout)
        self.setCentralWidget(self.viewer_panel)


        # Left: Image History
        self.image_history_dock = ImageHistoryDock(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.image_history_dock)
        self.image_history_dock.list_widget.itemClicked.connect(self.select_history_image)

        # Right: Captioner
        self.captioner_dock = CaptionerDock(self, DEFAULT_PROMPT)
        self.addDockWidget(Qt.RightDockWidgetArea, self.captioner_dock)
        self.captioner_dock.caption_btn.clicked.connect(self.generate_caption)

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        open_folder_action = QAction(QIcon(), "Open Folder", self)
        open_folder_action.triggered.connect(self.load_folder)
        toolbar.addAction(open_folder_action)
        save_caption_action = QAction(QIcon(), "Save Caption", self)
        save_caption_action.triggered.connect(self.save_caption)
        toolbar.addAction(save_caption_action)
        toolbar.addSeparator()
        # Zoom controls
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.graphics_view.zoom_in)
        toolbar.addAction(zoom_in_action)
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.graphics_view.zoom_out)
        toolbar.addAction(zoom_out_action)
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(self.graphics_view.reset_zoom)
        toolbar.addAction(reset_zoom_action)
        toolbar.addSeparator()
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        toolbar.addAction(about_action)


        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(open_folder_action)
        # Add batch caption menu option
        caption_all_action = QAction("Caption All Images in Folder", self)
        caption_all_action.triggered.connect(self.caption_all_images)
        file_menu.addAction(caption_all_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        help_menu = menubar.addMenu("Help")
        help_menu.addAction(about_action)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def load_model(self):
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map=0
            )
            self.llava_model.eval()
            self.status_bar.showMessage("Model loaded.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", f"Failed to load model: {e}")
            self.captioner_dock.caption_btn.setEnabled(False)
            self.status_bar.showMessage("Model load error.", 10000)

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", str(Path.home()))
        if folder_path:
            self.image_history_dock.clear_history()
            image_files = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.webp"):
                image_files.extend(Path(folder_path).glob(ext))
            image_files = sorted(image_files)
            if not image_files:
                QMessageBox.warning(self, "No Images", "No images found in the selected folder.")
                return
            for img_path in image_files:
                self.image_history_dock.add_image(str(img_path))
            # Auto-select the first image
            first_item = self.image_history_dock.list_widget.item(0)
            if first_item:
                self.image_history_dock.list_widget.setCurrentItem(first_item)
                path = first_item.data(Qt.UserRole)
                self.set_image(path)

    def set_image(self, file_path):
        # Save previous caption if changed
        if self._prev_image_path is not None:
            prev_caption = self.captioner_dock.caption_edit.toPlainText()
            if prev_caption != self._prev_caption:
                self._save_caption_to_file(self._prev_image_path, prev_caption)
        self.image_path = file_path
        self.image = Image.open(file_path)
        pixmap = QPixmap(file_path)
        self.graphics_view.set_image(pixmap)
        # Set the file name label below the image
        file_name = Path(file_path).name
        self.filename_label.setText(file_name)
        self.captioner_dock.caption_btn.setEnabled(True)
        # Load caption if exists
        txt_path = str(Path(file_path).with_suffix('.txt'))
        caption = ""
        if Path(txt_path).is_file():
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read()
        self.captioner_dock.caption_edit.setReadOnly(False)
        self.captioner_dock.caption_edit.setPlainText(caption)
        self._prev_image_path = file_path
        self._prev_caption = caption
        self.status_bar.showMessage(f"Loaded image: {file_path}", 5000)


    def select_history_image(self, item):
        # Get image path from item data
        path = item.data(Qt.UserRole)
        self.set_image(path)

    def generate_caption(self):
        if not self.image or not self.processor or not self.llava_model:
            QMessageBox.warning(self, "Missing Data", "Image or model not loaded.")
            return
        prompt = self.captioner_dock.prompt_input.text().strip() or DEFAULT_PROMPT
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]
        try:
            self.status_bar.showMessage("Generating caption...", 0)
            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[convo_string], images=[self.image], return_tensors="pt").to('cuda')
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
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
                )[0]
                generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
                caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                caption = caption.strip()
                self.captioner_dock.caption_edit.setPlainText(caption)
                # Auto-save caption as .txt next to image
                if self.image_path:
                    txt_path = str(Path(self.image_path).with_suffix('.txt'))
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                    self.status_bar.showMessage(f"Caption generated and saved: {txt_path}", 5000)
                else:
                    self.status_bar.showMessage("Caption generated.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Caption Error", f"Failed to generate caption: {e}")
            self.status_bar.showMessage("Caption error.", 10000)

    def save_caption(self):
        # Save current caption to file
        caption = self.captioner_dock.caption_edit.toPlainText()
        self._save_caption_to_file(self.image_path, caption)
        self._prev_caption = caption

    def _save_caption_to_file(self, image_path, caption):
        if not image_path:
            return
        txt_path = str(Path(image_path).with_suffix('.txt'))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        self.status_bar.showMessage(f"Caption saved: {txt_path}", 5000)

    def show_about(self):
        QMessageBox.information(self, "About JoyCaptioner", "JoyCaptioner\n\nImage Captioning App\nInspired by TagGUI\nhttps://github.com/fpgaminer/joycaption")

    def caption_all_images(self):
        images = []
        for i in range(self.image_history_dock.list_widget.count()):
            item = self.image_history_dock.list_widget.item(i)
            img_path = item.data(Qt.UserRole)
            images.append(img_path)
        if not images:
            QMessageBox.warning(self, "No Images", "No images loaded.")
            return
        txt_exists = any(Path(img).with_suffix('.txt').is_file() for img in images)
        if txt_exists:
            reply = QMessageBox.warning(self, "Overwrite Captions?", "This folder contains existing caption files. They will be overwritten if you continue. Proceed?", QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        # Start cancelable batch caption dialog
        self._batch_caption_dialog = BatchCaptionDialog(self, images, self.processor, self.llava_model, self.captioner_dock.prompt_input.text().strip() or DEFAULT_PROMPT)
        self._batch_caption_dialog.exec()
        self.status_bar.showMessage("Batch captioning complete.", 5000)

class BatchCaptionWorker(QObject):
    progress = Signal(int, int)  # current, total
    finished = Signal()
    error = Signal(str)

    def __init__(self, images, processor, llava_model, prompt):
        super().__init__()
        self.images = images
        self.processor = processor
        self.llava_model = llava_model
        self.prompt = prompt
        self.should_cancel = False

    def run(self):
        for idx, img_path in enumerate(self.images):
            if self.should_cancel:
                break
            try:
                image = Image.open(img_path)
                convo = [
                    {"role": "system", "content": "You are a helpful image captioner."},
                    {"role": "user", "content": self.prompt},
                ]
                convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
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
                    )[0]
                    generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
                    caption = self.processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    caption = caption.strip()
                    txt_path = str(Path(img_path).with_suffix('.txt'))
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
            except Exception as e:
                self.error.emit(f"Failed to caption {img_path}: {e}")
            self.progress.emit(idx + 1, len(self.images))
        self.finished.emit()

class BatchCaptionDialog(QDialog):
    def __init__(self, parent, images, processor, llava_model, prompt):
        super().__init__(parent)
        self.setWindowTitle("Batch Captioning")
        self.setModal(True)
        self.resize(400, 120)
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, len(images))
        layout.addWidget(self.progress_bar)
        self.cancel_button = QPushButton("Cancel", self)
        layout.addWidget(self.cancel_button)
        self.worker = BatchCaptionWorker(images, processor, llava_model, prompt)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.cancel_button.clicked.connect(self.cancel)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
    def on_progress(self, current, total):
        self.progress_bar.setValue(current)
    def cancel(self):
        self.worker.should_cancel = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling...")
    def on_finished(self):
        self.thread.quit()
        self.thread.wait()
        self.accept()
    def on_error(self, msg):
        print(msg)

class ImageGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = None
        self._zoom = 0
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(Qt.gray)

    def set_image(self, pixmap):
        self.scene().clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self.pixmap_item)
        self.setSceneRect(self.pixmap_item.boundingRect())
        self._zoom = 0
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        self._zoom += 1
        self.scale(1.25, 1.25)

    def zoom_out(self):
        self._zoom -= 1
        self.scale(0.8, 0.8)

    def reset_zoom(self):
        self._zoom = 0
        self.resetTransform()
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Modern Fusion style
    app.setStyle("Fusion")

    # Modern dark palette
    from PySide6.QtGui import QPalette, QColor, QFont
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(45, 45, 50))
    dark_palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 36))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 50))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    dark_palette.setColor(QPalette.Text, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.Button, QColor(60, 60, 65))
    dark_palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.Highlight, QColor(56, 140, 255))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(dark_palette)
    # Modern font
    app.setFont(QFont("Segoe UI", 10))
    # QSS Stylesheet for rounded corners, padding, hover
    app.setStyleSheet('''
        QWidget {
            border-radius: 6px;
            font-size: 10.5pt;
        }
        QMainWindow {
            background: #2d2d32;
        }
        QDockWidget {
            background: #23232a;
            border: none;
        }
        QListWidget {
            background: #23232a;
            border: none;
            padding: 6px;
        }
        QListWidget::item {
            border-radius: 8px;
            margin: 4px;
            padding: 8px;
        }
        QListWidget::item:selected {
            background: #388cff;
            color: white;
        }
        QPushButton {
            background: #388cff;
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background: #4ca3ff;
        }
        QLineEdit, QTextEdit {
            background: #23232a;
            color: #e0e0e0;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 6px;
        }
        QToolBar {
            background: #23232a;
            border: none;
        }
        QMenuBar {
            background: #23232a;
            color: #e0e0e0;
        }
        QMenuBar::item:selected {
            background: #388cff;
        }
        QStatusBar {
            background: #23232a;
            color: #e0e0e0;
            border: none;
        }
    ''')
    window = JoyCaptionMainWindow()
    window.show()
    sys.exit(app.exec())


    def load_model(self):
        try:
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_NAME, torch_dtype=torch.bfloat16, device_map=0
            )
            self.llava_model.eval()
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", f"Failed to load model: {e}")
            self.caption_btn.setEnabled(False)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path)
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.caption_btn.setEnabled(True)
            self.caption_edit.clear()
        else:
            self.caption_btn.setEnabled(False)

    def generate_caption(self):
        if not self.image or not self.processor or not self.llava_model:
            QMessageBox.warning(self, "Missing Data", "Image or model not loaded.")
            return
        prompt = self.prompt_input.text().strip() or DEFAULT_PROMPT
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]
        try:
            convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[convo_string], images=[self.image], return_tensors="pt").to('cuda')
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
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
                )[0]
                generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
                caption = self.processor.decode(generate_ids, skip_special_tokens=True)
                self.caption_edit.setPlainText(caption)
        except Exception as e:
            QMessageBox.critical(self, "Caption Error", f"Failed to generate caption: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JoyCaptionApp()
    window.show()
    sys.exit(app.exec())
