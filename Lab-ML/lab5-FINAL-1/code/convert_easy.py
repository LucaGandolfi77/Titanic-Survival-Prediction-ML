from PIL import Image
import os


def scala_immagine():
    print("=== SCALA IMMAGINE ===")

    # Immagine di input
    input_file = input("Inserisci il percorso dell'immagine: ").strip()

    if not os.path.exists(input_file):
        print("Errore: il file non esiste.")
        return

    # Carica immagine
    try:
        img = Image.open(input_file)
    except Exception as e:
        print(f"Errore nell'apertura dell'immagine: {e}")
        return

    print(f"\nDimensioni originali: {img.width} x {img.height} pixel")
    print(f"Formato originale: {img.format}")

    # Nuove dimensioni
    try:
        width = int(input("\nInserisci la nuova larghezza (pixel): "))
        height = int(input("Inserisci la nuova altezza (pixel): "))
    except ValueError:
        print("Errore: inserisci valori numerici.")
        return

    if width <= 0 or height <= 0:
        print("Errore: le dimensioni devono essere maggiori di 0.")
        return

    # Mantieni proporzioni?
    scelta = input(
        "\nVuoi mantenere le proporzioni? (s/n): "
    ).lower()

    if scelta == "s":
        # Calcola le dimensioni mantenendo il rapporto originale
        rapporto = img.width / img.height

        if width / height > rapporto:
            width = int(height * rapporto)
        else:
            height = int(width / rapporto)

        print(f"Dimensioni finali proporzionate: {width} x {height}")

    # Ridimensionamento
    img_resized = img.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    # Formato output
    print("\nFormati disponibili:")
    print("1 - PNG")
    print("2 - JPG")
    print("3 - WEBP")
    print("4 - BMP")

    formato_scelta = input("Scegli il formato: ")

    formati = {
        "1": ("PNG", ".png"),
        "2": ("JPEG", ".jpg"),
        "3": ("WEBP", ".webp"),
        "4": ("BMP", ".bmp")
    }

    if formato_scelta not in formati:
        print("Formato non valido.")
        return

    formato, estensione = formati[formato_scelta]

    # Nome file di output
    output_file = input(
        f"\nNome del file di output (senza estensione): "
    ).strip()

    if not output_file:
        output_file = "immagine_scalata"

    output_file += estensione

    # JPEG non supporta RGBA
    if formato == "JPEG" and img_resized.mode in ("RGBA", "LA", "P"):
        img_resized = img_resized.convert("RGB")

    # Salva
    img_resized.save(output_file, format=formato)

    print("\n=== OPERAZIONE COMPLETATA ===")
    print(f"Immagine originale: {img.width} x {img.height}")
    print(f"Immagine finale:    {width} x {height}")
    print(f"Formato:            {formato}")
    print(f"File salvato in:    {output_file}")


if __name__ == "__main__":
    scala_immagine()