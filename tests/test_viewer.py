from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from viewer_core import discover_images, find_duplicates, sort_images
from engine import (Page, Region, numbered_source, numbered_translations,
                    clean_image, add_translations, to_json_pages, from_json_pages)


def test_folder_navigation_natural_and_recursive(tmp_path):
    (tmp_path / 'nested').mkdir()
    for name in ('page10.png', 'page2.png', 'page1.png', 'note.txt'):
        (tmp_path / name).write_bytes(b'not checked here')
    (tmp_path / 'nested' / 'sub.png').write_bytes(b'img')
    assert [p.name for p in discover_images(tmp_path)] == ['page1.png', 'page2.png', 'page10.png']
    assert len(discover_images(tmp_path, recursive=True)) == 4
    assert [p.name for p in sort_images(discover_images(tmp_path), tmp_path)] == [
        'page1.png','page2.png','page10.png']


def test_exact_file_and_pixel_duplicates_across_formats_and_subfolders(tmp_path):
    (tmp_path / 'sub').mkdir()
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(40, 60, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    image = Image.fromarray(pixels, 'RGBA')
    first = tmp_path / 'one.png'
    image.save(first, 'PNG')
    copy = tmp_path / 'sub' / 'different_name.png'
    shutil.copyfile(first, copy)
    webp = tmp_path / 'same_pixels.webp'
    image.save(webp, 'WEBP', lossless=True)
    different = tmp_path / 'sub' / 'different.png'
    pixels[0,0,0] ^= 1
    Image.fromarray(pixels, 'RGBA').save(different)
    result = find_duplicates(discover_images(tmp_path, recursive=True))
    assert result.checked == 4
    assert not result.errors
    assert len(result.groups) == 1
    group = result.groups[0]
    assert set(group.paths) == {first, copy, webp}
    assert group.kind == 'pixels'
    assert different not in result.path_to_group()
    assert len(result.path_to_group()) == 3
    file_only = find_duplicates([first, copy])
    assert file_only.groups[0].kind == 'file'


def test_scan_cancellation_and_corrupt_file(tmp_path):
    good = tmp_path / 'good.png'
    Image.new('RGB', (12, 12), 'white').save(good)
    bad = tmp_path / 'bad.png'
    bad.write_text('not a png')
    cancelled = find_duplicates([good, bad], cancelled=lambda: True)
    assert cancelled.cancelled
    assert cancelled.checked == 0
    result = find_duplicates([good, bad])
    assert len(result.errors) == 1
    assert result.checked == 2


def test_area_actions_preserve_original_and_translation_list():
    original = Image.new('RGB',(250, 140),'white')
    from PIL import ImageDraw
    ImageDraw.Draw(original).text((30,40),'TEST',fill='black')
    keep = Region(1, (26,36,57,25), (25,35,85,40), 'original',
                  translated='must not be shown', action='keep', confirmed=True)
    hide = Region(2, (26,36,57,25), (25,35,85,40), 'erase',
                  action='hide', confirmed=True)
    translate = Region(3, (110,30,60,30), (110,30,110,70), 'translate',
                       translated='Hello!', action='translate', confirmed=True)
    assert clean_image(original, [keep]).tobytes() == original.tobytes()
    assert clean_image(original, [hide]).tobytes() != original.tobytes()
    output = add_translations(original,[keep, hide, translate])
    assert output.tobytes() != original.tobytes()
    pages = [Page('example.png',[keep, hide, translate])]
    assert numbered_source(pages) == '[0003] translate'
    assert numbered_translations(pages) == '[0003] Hello!'
    restored = from_json_pages(to_json_pages(pages))
    assert restored[0].regions[0].action == 'keep'
    assert restored[0].regions[2].confirmed is True
    # Backwards compatibility with JSON from version 1.
    old = [{'path':'test.png','regions':[{'id':1,'ink':[0,0,1,1],
                                          'box':[0,0,2,2],'source':'x'}]}]
    assert from_json_pages(old)[0].regions[0].action == 'translate'