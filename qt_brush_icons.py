"""Painter-based, scalable brush icons and cursor preview for the Qt editor."""
from __future__ import annotations
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QIcon, QPixmap, QPainter, QPen, QColor, QFont, QPolygonF


def brush_icon(kind: str) -> QIcon:
    pix=QPixmap(52,52)
    pix.fill(Qt.GlobalColor.transparent)
    p=QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    dark=QColor("#d6e7fa")
    accent=QColor("#477de0")
    p.setPen(QPen(dark,2.2,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap))
    p.setBrush(accent)
    if kind in ("marker","pixel","crayon"):
        p.drawRoundedRect(QRectF(21,7,13,30),3,3)
        p.setBrush(QColor("#8d9ce5"))
        p.drawRect(QRectF(21,19,13,13))
        p.setBrush(dark)
        p.drawPolygon(QPolygonF([QPointF(21,37),QPointF(34,37),QPointF(27.5,48)]))
        if kind=="crayon":
            p.setBrush(QColor("#e350a5"))
            p.drawRoundedRect(QRectF(21,7,13,14),2,2)
        if kind=="pixel":
            p.setBrush(QColor("#3268c4"))
            p.drawRect(QRectF(24,39,9,9))
    elif kind in ("calligraphy","pencil"):
        p.setBrush(QColor("#d5af70") if kind=="pencil" else QColor("#a3a8d8"))
        p.drawPolygon(QPolygonF([QPointF(21,5),QPointF(33,5),
                                  QPointF(32,34),QPointF(27,47),
                                  QPointF(22,34)]))
        p.setBrush(dark)
        p.drawEllipse(QRectF(25,37,4,4))
        if kind=="pencil":
            p.setBrush(QColor("#324c9e"))
            p.drawRect(QRectF(23,7,8,5))
    elif kind in ("oil","watercolor"):
        p.setBrush(QColor("#df9c47") if kind=="oil" else QColor("#4376ca"))
        p.drawRoundedRect(QRectF(22,7,12,26),2,2)
        p.setBrush(QColor("#f3c673") if kind=="oil" else QColor("#82cef2"))
        p.drawPolygon(QPolygonF([QPointF(22,33),QPointF(34,33),
                                  QPointF(30,49),QPointF(25,49)]))
    elif kind=="restore":
        p.setBrush(QColor("#ed869a"))
        p.drawPolygon(QPolygonF([QPointF(15,31),QPointF(28,13),
                                  QPointF(43,26),QPointF(30,44)]))
        p.setBrush(QColor("#eec4d2"))
        p.drawPolygon(QPolygonF([QPointF(15,31),QPointF(23,21),
                                  QPointF(38,35),QPointF(30,44)]))
    elif kind=="spray":
        p.setBrush(QColor("#ea991a"))
        p.drawRoundedRect(QRectF(21,17,15,30),4,4)
        p.drawRect(QRectF(24,9,9,7))
        p.setPen(QPen(dark,1.5))
        for x,y in ((11,10),(14,16),(9,22),(7,14),(16,5)):
            p.drawEllipse(QRectF(x,y,3,3))
    elif kind=="fill":
        p.setBrush(QColor("#889db9"))
        p.drawPolygon(QPolygonF([QPointF(15,21),QPointF(30,12),
                                  QPointF(43,27),QPointF(27,37)]))
        p.setBrush(QColor("#3299c6"))
        p.drawRoundedRect(QRectF(14,37,29,8),3,3)
    elif kind=="blur":
        # Three soft concentric rings: recognisable at toolbar and cursor size.
        p.setBrush(Qt.BrushStyle.NoBrush)
        for radius,alpha in ((18,65),(13,130),(8,245)):
            c=QColor("#93d8f2")
            c.setAlpha(alpha)
            p.setPen(QPen(c,3))
            p.drawEllipse(QPointF(27,27),radius,radius)
        p.setPen(QPen(QColor("#e9f6ff"),2))
        p.drawLine(27,18,27,36)
        p.drawLine(18,27,36,27)
    elif kind=="smudge":
        p.setBrush(QColor("#7e98ba"))
        p.drawRoundedRect(QRectF(12,29,30,7),3,3)
        p.setBrush(QColor("#c7d6de"))
        p.drawPolygon(QPolygonF([QPointF(35,30),QPointF(42,26),
                                  QPointF(43,38),QPointF(35,35)]))
    else:
        p.setBrush(QColor("#f3f7fb"))
        p.drawEllipse(QRectF(14,12,28,28))
        p.drawLine(9,26,47,26)
        p.drawLine(28,6,28,48)
    p.end()
    return QIcon(pix)


def picker_cursor_pixmap(color: QColor) -> QPixmap:
    """Pixel colour preview bubble, displayed above the ordinary arrow cursor."""
    pix=QPixmap(98,70)
    pix.fill(Qt.GlobalColor.transparent)
    p=QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing,True)
    p.setPen(QPen(QColor("#213b57"),2))
    p.setBrush(QColor("#ffffff"))
    p.drawRoundedRect(QRectF(3,2,92,57),14,14)
    p.setPen(QPen(QColor("#374c68"),1.5))
    p.setBrush(color)
    p.drawEllipse(QRectF(32,7,34,34))
    p.setPen(QColor("#1b2738"))
    f=QFont("Segoe UI",8)
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(6,39,86,18),Qt.AlignmentFlag.AlignCenter,color.name().upper())
    p.end()
    return pix
