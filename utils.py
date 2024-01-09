from PIL import Image, ImageDraw, ImageFont

def convertBoardToPIL(board):
    width, height = 84, 84

    blank_image = Image.new("RGB", (width, height), "white")
    
    blank_image.show()