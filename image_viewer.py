import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageViewer:
    MIN_GEOMETRY_ZOOM = 0.25
    MAX_GEOMETRY_ZOOM = 1.5
    MAX_GEOMETRY_PIXELS = 20_000_000

    def __init__(self, root):
        self.root = root
        self.root.title("Chuyển đổi hệ màu bằng OpenCV")
        self.root.geometry("1200x650")
        self.root.minsize(750, 400)

        self.images = {}
        self.photos = {}
        self.image_labels = {}
        self.bitwise_images = [None, None]
        self.bitwise_names = [None, None]
        self.bitwise_buttons = []
        self.save_source_image = None
        self.save_source_name = None
        self.brightness_source = None
        self.geometry_source = None
        self.geometry_current = None
        self.geometry_zoom = 1.0
        self.geometry_buttons = []

        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill="x")

        ttk.Button(
            toolbar,
            text="Chọn ảnh...",
            command=self.open_image,
        ).pack(side="left")

        self.status = ttk.Label(toolbar, text="Chưa chọn ảnh")
        self.status.pack(side="left", padx=12)

        # Add future features here; the navigation buttons are generated from this list.
        self.features = {
            "view": ("Xem ảnh", [("color", "Ảnh màu gốc")]),
            "convert": (
                "RGB / Grayscale / HSV",
                [("rgb", "RGB"), ("gray", "Grayscale"), ("hsv", "HSV")],
            ),
            "bitwise": (
                "Bitwise AND",
                [
                    ("and_image_1", "Ảnh thứ nhất"),
                    ("and_image_2", "Ảnh thứ hai"),
                    ("and_result", "Kết quả AND"),
                ],
            ),
            "save": ("Lưu ảnh PNG/JPEG/BMP", [("save_preview", "Xem trước ảnh")]),
            "brightness": (
                "Tăng độ sáng",
                [
                    ("brightness_original", "Ảnh gốc"),
                    ("brightness_result", "Ảnh đã tăng sáng"),
                ],
            ),
            "geometry": (
                "Biến đổi hình học",
                [("geometry_main", "Ảnh")],
            ),
        }
        self.feature_pages = {}
        self.feature_buttons = {}

        navigation = ttk.Frame(root, padding=(8, 0, 8, 8))
        navigation.pack(fill="x")
        for feature_key, (feature_name, _images) in self.features.items():
            button = tk.Button(
                navigation,
                text=feature_name,
                command=lambda key=feature_key: self.show_feature(key),
                relief="raised",
                borderwidth=2,
                padx=10,
                pady=5,
            )
            button.pack(side="left", padx=(0, 6))
            self.feature_buttons[feature_key] = button

        self.content = ttk.Frame(root, padding=(8, 0, 8, 8))
        self.content.pack(fill="both", expand=True)
        for feature_key, (_feature_name, images) in self.features.items():
            page = ttk.Frame(self.content)
            image_row = 1 if feature_key in (
                "bitwise",
                "save",
                "brightness",
                "geometry",
            ) else 0
            page.grid_rowconfigure(image_row, weight=1)

            if feature_key == "bitwise":
                controls = ttk.Frame(page, padding=(4, 4, 4, 8))
                controls.grid(row=0, column=0, columnspan=len(images), sticky="w")
                for index, button_text in enumerate(
                    ("Chọn ảnh thứ nhất...", "Chọn ảnh thứ hai...")
                ):
                    button = ttk.Button(
                        controls,
                        text=button_text,
                        command=lambda slot=index: self.load_bitwise_image(slot),
                    )
                    button.pack(side="left", padx=(0, 8))
                    self.bitwise_buttons.append(button)

                self.bitwise_status = ttk.Label(controls, text="Chọn hai ảnh để bắt đầu.")
                self.bitwise_status.pack(side="left", padx=4)
            elif feature_key == "save":
                controls = ttk.Frame(page, padding=(4, 4, 4, 8))
                controls.grid(row=0, column=0, sticky="w")

                ttk.Button(
                    controls,
                    text="Chọn ảnh cần lưu...",
                    command=self.load_image_for_saving,
                ).pack(side="left", padx=(0, 8))

                ttk.Label(controls, text="Định dạng:").pack(side="left", padx=(0, 4))
                self.save_format_var = tk.StringVar(value="PNG")
                ttk.Combobox(
                    controls,
                    textvariable=self.save_format_var,
                    values=("PNG", "JPEG", "BMP"),
                    state="readonly",
                    width=8,
                ).pack(side="left", padx=(0, 8))

                self.save_button = ttk.Button(
                    controls,
                    text="Lưu thành...",
                    command=self.save_image_as,
                    state="disabled",
                )
                self.save_button.pack(side="left", padx=(0, 8))
                self.save_status = ttk.Label(controls, text="Chưa chọn ảnh")
                self.save_status.pack(side="left")
            elif feature_key == "brightness":
                controls = ttk.Frame(page, padding=(4, 4, 4, 8))
                controls.grid(row=0, column=0, columnspan=len(images), sticky="w")

                ttk.Button(
                    controls,
                    text="Chọn ảnh...",
                    command=self.load_brightness_image,
                ).pack(side="left", padx=(0, 8))

                ttk.Label(controls, text="Giá trị cộng (0-255):").pack(
                    side="left", padx=(0, 4)
                )
                self.brightness_value_var = tk.StringVar(value="40")
                ttk.Spinbox(
                    controls,
                    from_=0,
                    to=255,
                    increment=1,
                    textvariable=self.brightness_value_var,
                    width=6,
                ).pack(side="left", padx=(0, 8))

                self.brightness_button = ttk.Button(
                    controls,
                    text="Tăng sáng",
                    command=self.apply_brightness,
                    state="disabled",
                )
                self.brightness_button.pack(side="left", padx=(0, 8))
                self.brightness_status = ttk.Label(controls, text="Chưa chọn ảnh")
                self.brightness_status.pack(side="left")
            elif feature_key == "geometry":
                controls = ttk.Frame(page, padding=(4, 4, 4, 8))
                controls.grid(row=0, column=0, columnspan=len(images), sticky="w")

                ttk.Button(
                    controls,
                    text="Chọn ảnh...",
                    command=self.load_geometry_image,
                ).pack(side="left", padx=(0, 8))

                for button_text, command in (
                    ("Xoay 90 độ", self.rotate_geometry),
                    ("Dịch trái 50 px", lambda: self.translate_geometry(-50)),
                    ("Dịch phải 50 px", lambda: self.translate_geometry(50)),
                    ("Phóng to 1.5x", lambda: self.scale_geometry(1.5)),
                    ("Thu nhỏ 1.5x", lambda: self.scale_geometry(1 / 1.5)),
                    ("Đặt lại", self.reset_geometry),
                ):
                    button = ttk.Button(
                        controls,
                        text=button_text,
                        command=command,
                        state="disabled",
                    )
                    button.pack(side="left", padx=(0, 6))
                    self.geometry_buttons.append(button)

                self.geometry_status = ttk.Label(controls, text="Chưa chọn ảnh")
                self.geometry_status.pack(side="left")

            for column, (image_key, title) in enumerate(images):
                page.grid_columnconfigure(column, weight=1, uniform=feature_key)
                panel = ttk.LabelFrame(page, text=title, padding=6)
                panel.grid(row=image_row, column=column, sticky="nsew", padx=4)

                image_label = ttk.Label(panel, anchor="center")
                image_label.pack(fill="both", expand=True)
                image_label.bind(
                    "<Configure>",
                    lambda _event, key=image_key: self.resize_display(key),
                )
                self.image_labels[image_key] = image_label
            self.feature_pages[feature_key] = page

        self.show_feature("view")

    def show_feature(self, feature_key):
        for page in self.feature_pages.values():
            page.pack_forget()
        self.feature_pages[feature_key].pack(fill="both", expand=True)
        for key, button in self.feature_buttons.items():
            button.config(relief="sunken" if key == feature_key else "raised")

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[
                ("Tệp ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not path:
            return

        try:
            # fromfile hỗ trợ đường dẫn Windows có dấu hoặc ký tự Unicode.
            encoded = np.fromfile(path, dtype=np.uint8)
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("OpenCV không đọc được tệp ảnh này.")

            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            self.images = {
                "color": Image.fromarray(image_rgb),
                "rgb": Image.fromarray(image_rgb),
                "gray": Image.fromarray(image_gray),
                # Hiển thị trực tiếp mảng HSV (H, S, V được coi như 3 kênh R, G, B)
                # -> không chuyển ngược về RGB nữa nên màu sẽ khác ảnh gốc.
                "hsv": Image.fromarray(image_hsv),
            }

            height, width = image_bgr.shape[:2]
            self.status.config(
                text=f"{os.path.basename(path)} ({width} x {height})"
            )
            self.root.title(f"Chuyển đổi hệ màu - {os.path.basename(path)}")
            for key in self.images:
                self.resize_display(key)

        except Exception as error:
            messagebox.showerror("Không thể mở ảnh", str(error))

    def load_bitwise_image(self, index):
        path = filedialog.askopenfilename(
            title=f"Chọn ảnh thứ {index + 1}",
            filetypes=[
                ("Tệp ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not path:
            return

        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV không đọc được tệp ảnh này.")

            self.bitwise_images[index] = image
            self.bitwise_names[index] = os.path.basename(path)
            image_key = f"and_image_{index + 1}"
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images[image_key] = Image.fromarray(image_rgb)
            self.bitwise_buttons[index].config(
                text=f"Đổi ảnh thứ {index + 1}..."
            )
            self.resize_display(image_key)
            self.update_bitwise_result()

        except Exception as error:
            messagebox.showerror("Không thể mở ảnh", str(error))

    def update_bitwise_result(self):
        if any(image is None for image in self.bitwise_images):
            chosen = [
                f"ảnh {index + 1}: {name}"
                for index, name in enumerate(self.bitwise_names)
                if name is not None
            ]
            if chosen:
                self.bitwise_status.config(
                    text=f"Đã chọn {', '.join(chosen)}. Hãy chọn ảnh còn lại."
                )
            return

        image_1, image_2 = self.bitwise_images
        height, width = image_1.shape[:2]
        resized = image_1.shape[:2] != image_2.shape[:2]
        if resized:
            image_2 = cv2.resize(image_2, (width, height))

        result = cv2.bitwise_and(image_1, image_2)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self.images["and_result"] = Image.fromarray(result_rgb)
        self.resize_display("and_result")

        if resized:
            self.bitwise_status.config(
                text=f"Ảnh thứ hai đã đổi kích thước thành {width} x {height}."
            )
        else:
            self.bitwise_status.config(
                text=f"Đã áp dụng AND trên hai ảnh {width} x {height}."
            )

    def load_image_for_saving(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh cần lưu",
            filetypes=[
                ("Tệp ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not path:
            return

        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV không đọc được tệp ảnh này.")

            if image.ndim == 2:
                self.save_source_image = Image.fromarray(image)
            elif image.shape[2] == 4:
                rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                self.save_source_image = Image.fromarray(rgba)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.save_source_image = Image.fromarray(rgb)

            self.save_source_name = os.path.basename(path)
            self.images["save_preview"] = self.save_source_image.copy()
            self.save_button.config(state="normal")
            self.save_status.config(text=self.save_source_name)
            self.resize_display("save_preview")
        except Exception as error:
            messagebox.showerror("Không thể mở ảnh", str(error))

    def save_image_as(self):
        if self.save_source_image is None:
            return

        format_name = self.save_format_var.get()
        extensions = {"PNG": ".png", "JPEG": ".jpg", "BMP": ".bmp"}
        extension = extensions[format_name]
        initial_name = os.path.splitext(self.save_source_name)[0] + extension
        path = filedialog.asksaveasfilename(
            title="Lưu ảnh thành",
            initialfile=initial_name,
            defaultextension=extension,
            filetypes=[(format_name, f"*{extension}")],
        )
        if not path:
            return

        base, current_extension = os.path.splitext(path)
        if current_extension.lower() != extension:
            path = base + extension

        try:
            image = self.save_source_image
            if format_name == "JPEG":
                if image.mode == "RGBA":
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    image = background
                elif image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
            elif format_name == "BMP" and image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            image.save(path, format=format_name)
            self.save_status.config(text=f"Đã lưu: {os.path.basename(path)}")
        except Exception as error:
            messagebox.showerror("Không thể lưu ảnh", str(error))

    def load_brightness_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh cần tăng độ sáng",
            filetypes=[
                ("Tệp ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not path:
            return

        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV không đọc được tệp ảnh này.")

            self.brightness_source = image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images["brightness_original"] = Image.fromarray(image_rgb)
            self.brightness_button.config(state="normal")
            self.brightness_status.config(text=os.path.basename(path))
            self.resize_display("brightness_original")
            self.apply_brightness()
        except Exception as error:
            messagebox.showerror("Không thể mở ảnh", str(error))

    def apply_brightness(self):
        if self.brightness_source is None:
            return

        try:
            value = int(self.brightness_value_var.get())
            if not 0 <= value <= 255:
                raise ValueError("Giá trị cộng phải nằm trong khoảng 0 đến 255.")
        except ValueError as error:
            messagebox.showerror("Giá trị không hợp lệ", str(error))
            return

        addition = np.full_like(self.brightness_source, value)
        result = cv2.add(self.brightness_source, addition)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self.images["brightness_result"] = Image.fromarray(result_rgb)
        self.brightness_status.config(text=f"Đã cộng {value} vào từng kênh pixel.")
        self.resize_display("brightness_result")

    def load_geometry_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh cần biến đổi",
            filetypes=[
                ("Tệp ảnh", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if not path:
            return

        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV không đọc được tệp ảnh này.")

            self.geometry_source = image
            self.geometry_current = image.copy()
            self.geometry_zoom = 1.0
            for button in self.geometry_buttons:
                button.config(state="normal")
            self.update_geometry_display(os.path.basename(path))
        except Exception as error:
            messagebox.showerror("Không thể mở ảnh", str(error))

    def rotate_geometry(self):
        if self.geometry_current is None:
            return
        self.geometry_current = cv2.rotate(
            self.geometry_current,
            cv2.ROTATE_90_CLOCKWISE,
        )
        self.update_geometry_display("Đã xoay 90 độ theo chiều kim đồng hồ.")

    def translate_geometry(self, horizontal_offset):
        if self.geometry_current is None:
            return
        height, width = self.geometry_current.shape[:2]
        translation = np.float32([[1, 0, horizontal_offset], [0, 1, 0]])
        self.geometry_current = cv2.warpAffine(
            self.geometry_current,
            translation,
            (width, height),
        )
        direction = "phải" if horizontal_offset > 0 else "trái"
        self.update_geometry_display(f"Đã dịch {direction} {abs(horizontal_offset)} px.")

    def scale_geometry(self, factor):
        if self.geometry_current is None:
            return

        new_zoom = self.geometry_zoom * factor
        if not self.MIN_GEOMETRY_ZOOM <= new_zoom <= self.MAX_GEOMETRY_ZOOM:
            self.geometry_status.config(
                text=(
                    f"Đã đạt giới hạn thu phóng "
                    f"{self.MIN_GEOMETRY_ZOOM}x-{self.MAX_GEOMETRY_ZOOM}x."
                )
            )
            return

        height, width = self.geometry_current.shape[:2]
        resize_factor = new_zoom / self.geometry_zoom
        new_width = max(1, round(width * resize_factor))
        new_height = max(1, round(height * resize_factor))
        if new_width * new_height > self.MAX_GEOMETRY_PIXELS:
            self.geometry_status.config(
                text="Đã đạt giới hạn kích thước ảnh để tránh quá tải bộ nhớ."
            )
            return

        self.geometry_current = cv2.resize(
            self.geometry_current,
            (new_width, new_height),
            interpolation=(
                cv2.INTER_LINEAR if resize_factor > 1 else cv2.INTER_AREA
            ),
        )
        self.geometry_zoom = new_zoom
        action = "phóng to" if factor > 1 else "thu nhỏ"
        self.update_geometry_display(f"Đã {action} 1.5 lần.")

    def reset_geometry(self):
        if self.geometry_source is None:
            return
        self.geometry_current = self.geometry_source.copy()
        self.geometry_zoom = 1.0
        self.update_geometry_display("Đã đặt lại ảnh ban đầu.")

    def update_geometry_display(self, message):
        height, width = self.geometry_current.shape[:2]
        current_rgb = cv2.cvtColor(self.geometry_current, cv2.COLOR_BGR2RGB)
        self.images["geometry_main"] = Image.fromarray(current_rgb)
        self.geometry_status.config(
            text=(
                f"{message} Kích thước: {width} x {height}. "
                f"Thu phóng: {self.geometry_zoom:.2f}x."
            )
        )
        self.resize_display("geometry_main")

    def resize_display(self, key):
        if key not in self.images:
            return

        image_label = self.image_labels[key]
        label_width = image_label.winfo_width()
        label_height = image_label.winfo_height()
        if label_width <= 1 or label_height <= 1:
            return

        max_width = label_width - 12
        max_height = label_height - 12

        preview = self.images[key].copy()
        preview.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS,
        )

        self.photos[key] = ImageTk.PhotoImage(preview)
        image_label.config(image=self.photos[key])


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()