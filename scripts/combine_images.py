from PIL import Image
from pathlib import Path

#dateiname:

filename = "amp11_combined.png"
#Pfad zu  den Bildern
image_dir = Path(__file__).resolve().parents[1] / 'results' / 'plots'
# Sicherstellen, dass der Pfad existiert
image_dir.mkdir(parents=True, exist_ok=True)
# Dateinamen der Bilder
stem = "Sub8_Kinematics_T3"
image1_path = image_dir / (stem + "_10_freeze.png")
image2_path = image_dir / (stem + "_10_amp_modes.png")
# Lade die Bilder
img1 = Image.open(image1_path)
img2 = Image.open(image2_path)

# Ermittle Breiten und Höhen
width1, height1 = img1.size
width2, height2 = img2.size

# Gesamtbreite und maximale Höhe berechnen
total_width = width1 + width2
max_height = max(height1, height2)

# Neues Bild mit weißem Hintergrund erstellen (oder transparent, falls benötigt)
mode = 'RGBA' if 'A' in (img1.mode + img2.mode) else 'RGB'
background = (255, 255, 255, 0) if mode == 'RGBA' else (255, 255, 255)
combined = Image.new(mode, (total_width, max_height), background)

# Bilder nebeneinander einfügen
combined.paste(img1, (0, 0))
combined.paste(img2, (width1, 0))

# Ergebnis speichern
combined.save(image_dir / filename)
print(f"Combined image saved as {filename}")