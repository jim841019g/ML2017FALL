import sys
from PIL import Image

testing_image = sys.argv[1]
im = Image.open(testing_image)
width,height = im.size
output = Image.new('RGB',(width,height),'white')

for x in range(width):
    for y in range(height):
        rgb = im.getpixel((x, y))
        outrgb =  (int(rgb[0]/2) , int(rgb[1]/2) , int(rgb[2]/2))
        output.putpixel((x, y), outrgb)

output.save('Q2.jpg')
