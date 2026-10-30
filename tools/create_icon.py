"""Produce a Windows icon reproducibly with Pillow (no binary assets needed)."""
from pathlib import Path
from PIL import Image, ImageDraw

output = Path(__file__).resolve().parents[1] / 'assets' / 'manga.ico'
output.parent.mkdir(parents=True, exist_ok=True)
size = 256
im = Image.new('RGBA', (size, size), '#172334')
d = ImageDraw.Draw(im)
d.rounded_rectangle((7, 7, 249, 249), radius=47, fill='#234a76')
d.rounded_rectangle((27, 27, 229, 229), radius=40, fill='#2b80cf')
d.rounded_rectangle((48, 47, 207, 179), radius=34, fill='#f8fbff')
d.polygon([(84, 170), (83, 208), (126, 174)], fill='#f8fbff')
d.rounded_rectangle((73, 81, 183, 95), radius=7, fill='#38618f')
d.rounded_rectangle((73, 111, 171, 125), radius=7, fill='#38618f')
d.rounded_rectangle((73, 141, 143, 155), radius=7, fill='#38618f')
im.save(output, format='ICO', sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(output)