"""Independent undo/redo commands for raster and editable text.

A raster command stores only the changed pixel rectangle; a text command
stores only text layers. This prevents undoing a cleanup from rolling back
an unrelated text edit. A single chronological stack contains both kinds.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from PIL import Image, ImageChops


@dataclass
class EditorCommand:
    kind: str
    label: str
    before: object
    after: object
    box: tuple[int,int,int,int] | None = None
    before_index: int = -1
    after_index: int = -1

    def apply(self, image, layers, *, undo: bool):
        source = self.before if undo else self.after
        if self.kind == "raster":
            out = image.copy()
            out.paste(source, self.box[:2])
            return out, layers, None
        if self.kind == "text":
            return image, deepcopy(source), (
                self.before_index if undo else self.after_index)
        raise ValueError("Unsupported history command: " + self.kind)


def raster_command(before: Image.Image, after: Image.Image,
                   label: str = "Изменение изображения") -> EditorCommand | None:
    if before.size != after.size:
        raise ValueError("Raster edit must preserve image dimensions")
    # RGB bounding box is correct even if RGBA alpha is unchanged.
    left = before.convert("RGBA")
    right = after.convert("RGBA")
    rgb_diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
    alpha_diff = ImageChops.difference(left.getchannel("A"), right.getchannel("A"))
    bounds = rgb_diff.getbbox()
    a = alpha_diff.getbbox()
    if a is not None:
        if bounds is None:
            bounds = a
        else:
            bounds = (min(bounds[0],a[0]), min(bounds[1],a[1]),
                      max(bounds[2],a[2]), max(bounds[3],a[3]))
    if bounds is None:
        return None
    return EditorCommand("raster",label,before.crop(bounds),after.crop(bounds),
                         box=bounds)


def text_command(before, after, before_index=-1, after_index=-1,
                 label="Изменение текста") -> EditorCommand | None:
    if before == after:
        return None
    return EditorCommand("text",label,deepcopy(before),deepcopy(after),
                         before_index=before_index,after_index=after_index)


class CommandHistory:
    def __init__(self, limit=60):
        self.steps: list[EditorCommand] = []
        self.future: list[EditorCommand] = []
        self.limit=limit

    def __len__(self):
        return len(self.steps)

    def push(self, step:EditorCommand | None):
        if step is None:
            return False
        self.steps.append(step)
        if len(self.steps)>self.limit:
            del self.steps[0]
        self.future.clear()
        return True

    def undo(self, image, layers):
        if not self.steps:
            return image,layers,None
        step=self.steps.pop()
        self.future.append(step)
        return step.apply(image,layers,undo=True)

    def redo(self, image, layers):
        if not self.future:
            return image,layers,None
        step=self.future.pop()
        self.steps.append(step)
        return step.apply(image,layers,undo=False)
