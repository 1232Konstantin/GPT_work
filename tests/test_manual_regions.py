from PIL import Image,ImageDraw
from engine import Page,Region, to_json_pages,from_json_pages,clean_image,add_translations
from manual_regions import suggest_boundary,inner_text_box,poly_mask,LocalRecognition

def test_blank_contour_fallback():
    im=Image.new('RGB',(200,200),'white')
    assert suggest_boundary(im,(50,50,40,60))==[]

def test_local_contour_rough_balloon():
    im=Image.new('RGB',(250,260),'#ababab')
    d=ImageDraw.Draw(im)
    d.ellipse((30,25,210,245),fill='white',outline='black',width=3)
    polygon=suggest_boundary(im,(100,110,30,45))
    if polygon:
        box=inner_text_box(polygon,(100,110,30,45),im.size)
        assert box[2]>0 and box[3]>0

def test_polygon_clipping_and_json_roundtrip():
    im=Image.new('RGB',(180,160),'white')
    r=Region(1,(66,55,20,20),(20,20,120,120),source='かな',translated='Hello',
             polygon=[(40,35),(125,35),(125,135),(40,135)],color='#5a2534',bold=True)
    page=Page('test', [r])
    recovered=from_json_pages(to_json_pages([page]))[0].regions[0]
    assert recovered.color=='#5a2534' and recovered.bold is True
    assert len(recovered.polygon)==4
    output=add_translations(clean_image(im,[r]),[r])
    assert output.size==im.size

def test_keep_does_not_change_image():
    im=Image.new('RGB',(100,100),'white')
    r=Region(1,(30,30,20,20),(20,20,40,40),action='keep')
    assert clean_image(im,[r]).tobytes()==im.tobytes()