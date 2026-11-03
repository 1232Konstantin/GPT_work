"""Neutral, crisp vector Qt toolbar icons sized for a dark desktop theme."""
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QIcon, QPixmap, QPainter, QPen, QPainterPath, QPolygonF, QFont


def toolbar_icon(name, color="#e7edf6"):
    px=QPixmap(64,64)
    px.fill(Qt.GlobalColor.transparent)
    painter=QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing,True)
    c=QColor(color)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(c,5,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap,
                        Qt.PenJoinStyle.RoundJoin))
    if name in ("undo","redo"):
        painter.save()
        if name=="redo":
            painter.translate(64,0)
            painter.scale(-1,1)
        path=QPainterPath(QPointF(19,20))
        path.cubicTo(QPointF(34,13),QPointF(49,24),QPointF(47,39))
        path.cubicTo(QPointF(46,51),QPointF(31,53),QPointF(27,48))
        painter.drawPath(path)
        painter.drawLine(21,19,34,10)
        painter.drawLine(21,19,32,31)
        painter.restore()
    elif name=="copy":
        painter.drawRoundedRect(15,18,32,34,3,3)
        painter.drawLine(23,12,51,12)
        painter.drawLine(51,12,51,42)
    elif name=="paste":
        painter.drawRoundedRect(13,16,38,39,3,3)
        painter.drawRoundedRect(22,8,20,12,3,3)
        painter.drawLine(23,32,41,32)
        painter.drawLine(23,41,39,41)
    elif name=="select":
        painter.drawRect(12,12,40,40)
    elif name=="cleanup":
        painter.drawRoundedRect(15,17,34,26,5,5)
        painter.drawLine(23,38,38,22)
    elif name=="zoom":
        painter.drawEllipse(12,11,30,30)
        painter.drawLine(37,38,53,54)
    elif name=="text":
        font=QFont("Segoe UI",30)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(5,4,54,56),Qt.AlignmentFlag.AlignCenter,"Т")
    elif name=="pan":
        painter.drawLine(22,34,22,19)
        painter.drawLine(22,19,27,13)
        painter.drawLine(27,13,31,13)
        painter.drawLine(31,13,31,30)
        painter.drawLine(31,30,34,12)
        painter.drawLine(34,12,40,12)
        painter.drawLine(40,12,39,32)
        painter.drawLine(39,32,47,26)
        painter.drawLine(47,26,52,29)
        painter.drawLine(52,29,45,48)
        painter.drawLine(45,48,28,53)
        painter.drawLine(28,53,12,36)
        painter.drawLine(12,36,14,30)
        painter.drawLine(14,30,22,34)
    else:
        painter.drawEllipse(19,19,26,26)
    painter.end()
    return QIcon(px)
