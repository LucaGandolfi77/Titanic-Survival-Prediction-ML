import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

# Libreria per il drag & drop
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    print("Errore: installa tkinterdnd2 con:")
    print("pip install tkinterdnd2")
    exit()


class ImageScalerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Scaler")
        self.root.geometry("800x650")
        self.root.minsize(700, 550)

        self.image = None
        self.preview_image = None
        self.input_path = None

        # =========================
        # TITOLO
        # =========================

        title = tk.Label(
            root,
            text="Image Scaler",
            font=("Arial", 22, "bold")
        )
        title.pack(pady=15)

        # =========================
        # AREA DRAG & DROP
        # =========================

        self.drop_area = tk.Frame(
            root,
            height=180,
            bd=2,
            relief="groove"
        )
        self.drop_area.pack(
            fill="x",
            padx=30,
            pady=10
        )

        self.drop_label = tk.Label(
            self.drop_area,
            text="Trascina qui un'immagine\n\n"
                 "oppure clicca per selezionarla",
            font=("Arial", 14),
            justify="center"
        )

        self.drop_label.pack(
            expand=True,
            fill="both"
        )

        # Drag & drop
        self.drop_area.drop_target_register(DND_FILES)
        self.drop_area.dnd_bind(
            "<<Drop>>",
            self.drop_image
        )

        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind(
            "<<Drop>>",
            self.drop_image
        )

        # Click per scegliere file
        self.drop_area.bind(
            "<Button-1>",
            self.select_image
        )

        self.drop_label.bind(
            "<Button-1>",
            self.select_image
        )

        # =========================
        # INFORMAZIONI IMMAGINE
        # =========================

        self.info_label = tk.Label(
            root,
            text="Nessuna immagine selezionata",
            font=("Arial", 10)
        )

        self.info_label.pack(pady=5)

        # =========================
        # DIMENSIONI
        # =========================

        size_frame = tk.LabelFrame(
            root,
            text="Dimensioni",
            padx=15,
            pady=10
        )

        size_frame.pack(
            fill="x",
            padx=30,
            pady=10
        )

        tk.Label(
            size_frame,
            text="Larghezza:"
        ).grid(row=0, column=0, padx=5, pady=5)

        self.width_var = tk.StringVar()

        self.width_entry = tk.Entry(
            size_frame,
            textvariable=self.width_var,
            width=10
        )

        self.width_entry.grid(
            row=0,
            column=1,
            padx=5
        )

        tk.Label(
            size_frame,
            text="px"
        ).grid(row=0, column=2)

        tk.Label(
            size_frame,
            text="Altezza:"
        ).grid(row=0, column=3, padx=5)

        self.height_var = tk.StringVar()

        self.height_entry = tk.Entry(
            size_frame,
            textvariable=self.height_var,
            width=10
        )

        self.height_entry.grid(
            row=0,
            column=4,
            padx=5
        )

        tk.Label(
            size_frame,
            text="px"
        ).grid(row=0, column=5)

        # Mantieni proporzioni
        self.keep_ratio = tk.BooleanVar(value=True)

        ratio_check = tk.Checkbutton(
            size_frame,
            text="Mantieni proporzioni",
            variable=self.keep_ratio
        )

        ratio_check.grid(
            row=1,
            column=0,
            columnspan=6,
            pady=8
        )

        # =========================
        # FORMATO
        # =========================

        format_frame = tk.LabelFrame(
            root,
            text="Formato output",
            padx=15,
            pady=10
        )

        format_frame.pack(
            fill="x",
            padx=30,
            pady=10
        )

        self.format_var = tk.StringVar(
            value="PNG"
        )

        formats = [
            "PNG",
            "JPEG",
            "WEBP",
            "BMP"
        ]

        self.format_combo = ttk.Combobox(
            format_frame,
            textvariable=self.format_var,
            values=formats,
            state="readonly",
            width=15
        )

        self.format_combo.pack()

        # =========================
        # PREVIEW
        # =========================

        preview_frame = tk.LabelFrame(
            root,
            text="Anteprima",
            padx=10,
            pady=10
        )

        preview_frame.pack(
            fill="both",
            expand=True,
            padx=30,
            pady=10
        )

        self.preview_label = tk.Label(
            preview_frame,
            text="Nessuna anteprima"
        )

        self.preview_label.pack(
            expand=True
        )

        # =========================
        # PULSANTE SALVA
        # =========================

        self.save_button = tk.Button(
            root,
            text="SCALA E SALVA",
            font=("Arial", 12, "bold"),
            command=self.save_image,
            state="disabled",
            padx=20,
            pady=8
        )

        self.save_button.pack(
            pady=15
        )

    # ==================================================
    # SELEZIONE IMMAGINE
    # ==================================================

    def select_image(self, event=None):

        file_path = filedialog.askopenfilename(
            title="Seleziona un'immagine",
            filetypes=[
                (
                    "Immagini",
                    "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tiff"
                ),
                ("Tutti i file", "*.*")
            ]
        )

        if file_path:
            self.load_image(file_path)

    # ==================================================
    # DRAG & DROP
    # ==================================================

    def drop_image(self, event):

        # tkinterdnd2 restituisce il percorso tra parentesi
        file_path = event.data

        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]

        # Se vengono trascinati più file,
        # prendiamo il primo
        if "} {" in file_path:
            file_path = file_path.split("} {")[0]
            file_path = file_path.strip("{}")

        self.load_image(file_path)

    # ==================================================
    # CARICA IMMAGINE
    # ==================================================

    def load_image(self, file_path):

        try:

            image = Image.open(file_path)

            # Convertiamo GIF / P / ecc.
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")

            self.image = image
            self.input_path = file_path

            # Dimensioni originali
            width = image.width
            height = image.height

            self.width_var.set(str(width))
            self.height_var.set(str(height))

            self.info_label.config(
                text=(
                    f"Immagine: {os.path.basename(file_path)}   |   "
                    f"Dimensioni: {width} × {height} px   |   "
                    f"Formato: {image.format}"
                )
            )

            # Mostra anteprima
            self.show_preview()

            self.save_button.config(
                state="normal"
            )

        except Exception as e:

            messagebox.showerror(
                "Errore",
                f"Impossibile aprire l'immagine:\n\n{e}"
            )

    # ==================================================
    # ANTEPRIMA
    # ==================================================

    def show_preview(self):

        if self.image is None:
            return

        preview = self.image.copy()

        # Dimensione massima anteprima
        max_width = 500
        max_height = 250

        preview.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS
        )

        self.preview_image = ImageTk.PhotoImage(
            preview
        )

        self.preview_label.config(
            image=self.preview_image,
            text=""
        )

    # ==================================================
    # RIDIMENSIONAMENTO
    # ==================================================

    def resize_image(self):

        if self.image is None:
            return None

        try:

            width = int(
                self.width_var.get()
            )

            height = int(
                self.height_var.get()
            )

        except ValueError:

            messagebox.showerror(
                "Errore",
                "Larghezza e altezza devono essere numeri interi."
            )

            return None

        if width <= 0 or height <= 0:

            messagebox.showerror(
                "Errore",
                "Le dimensioni devono essere maggiori di zero."
            )

            return None

        # Mantieni proporzioni
        if self.keep_ratio.get():

            original_ratio = (
                self.image.width /
                self.image.height
            )

            requested_ratio = (
                width / height
            )

            if requested_ratio > original_ratio:

                width = int(
                    height * original_ratio
                )

            else:

                height = int(
                    width / original_ratio
                )

        resized = self.image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )

        return resized, width, height

    # ==================================================
    # SALVA
    # ==================================================

    def save_image(self):

        result = self.resize_image()

        if result is None:
            return

        resized, width, height = result

        format_name = self.format_var.get()

        extensions = {
            "PNG": ".png",
            "JPEG": ".jpg",
            "WEBP": ".webp",
            "BMP": ".bmp"
        }

        extension = extensions[format_name]

        # Nome suggerito
        original_name = os.path.splitext(
            os.path.basename(self.input_path)
        )[0]

        suggested_name = (
            f"{original_name}_{width}x{height}"
        )

        output_path = filedialog.asksaveasfilename(
            title="Salva immagine",
            initialfile=suggested_name + extension,
            defaultextension=extension,
            filetypes=[
                (format_name, "*" + extension)
            ]
        )

        if not output_path:
            return

        try:

            # JPEG non supporta RGBA
            if (
                format_name == "JPEG"
                and resized.mode == "RGBA"
            ):
                resized = resized.convert("RGB")

            resized.save(
                output_path,
                format=format_name
            )

            messagebox.showinfo(
                "Completato",
                f"Immagine salvata correttamente!\n\n"
                f"Dimensioni: {width} × {height} px\n"
                f"Formato: {format_name}\n\n"
                f"{output_path}"
            )

        except Exception as e:

            messagebox.showerror(
                "Errore",
                f"Impossibile salvare l'immagine:\n\n{e}"
            )


# ======================================================
# AVVIO PROGRAMMA
# ======================================================

if __name__ == "__main__":

    root = TkinterDnD.Tk()

    app = ImageScalerApp(root)

    root.mainloop()
