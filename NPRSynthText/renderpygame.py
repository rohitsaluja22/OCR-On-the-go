# Pygame setup and create root window
#https://www.pygame.org/docs/ref/font.html#pygame.font.Font
#https://www.pygame.org/docs/ref/freetype.html#pygame.freetype.Font
# -*- coding: utf-8 -*-
import time
import pygame
import pygame.freetype

pygame.font.init()
pygame.freetype.init()

screen = pygame.display.set_mode((320, 200))
empty = pygame.Surface((320, 200))
'''101Lohit/Lohit14042007.ttf
102Mukti/Mukti1p99PR.ttf
1SolaimanLipi/SolaimanLipi.ttf
2Nikosh/Nikosh.ttf
3AmarBangla/AmarBanglaBold.ttf
3AmarBangla/AmarBangla.ttf
4SutonnyMJ/SutonnyOMJ.ttf
no upto 9

https://www.omicronlab.com/bangla-fonts.html
kalpurush.ttf 47
Siyamrupali.ttf 48
AdorshoLipi_20-07-2007.ttf 46
AponaLohit.ttf 50
Bangla.ttf 37
BenSenHandwriting.ttf 35
BenSen.ttf 35
Nikosh.ttf 35
SolaimanLipi_20-04-07.ttf 40
akaashnormal.ttf 33
Lohit_14-04-2007.ttf 50
mitra.ttf 44
Mukti_1.99_PR.ttf 39
muktinarrow.ttf 50
NikoshBAN.ttf 35
NikoshGrameen.ttf 43
NikoshLightBan.ttf 35
NikoshLight.ttf 35
sagarnormal.ttf 40
'''
DATA_PATH = "/home/rohit/src2/1SceneTextBangla/data/fonts/omicronlab/"
#unistr = "♛"
#font_file = pygame.font.match_font("Shobhika")  # Select and Shobhika Sanskrit_2003
#print(font_file)
font_file = DATA_PATH + "sagarnormal.ttf"#"/home/rohit/src2/1SceneTextBangla/data/fonts/9ChitraMJ/ChitraMJ-BoldItalic.ttf"
print(font_file)
font = pygame.freetype.Font(font_file,30)
print(font.get_sized_height())
#font = pygame.font.Font(font_file, 30)          # open the font
#writing,dg = font.render(chr(0x0915),(0, 0, 0),(255, 255, 255))
#writing,dg = font.render("द",(0, 0, 0),(255, 255, 255))  # Render text on a surface
rect = font.render_to(empty,(40,40),"আচওয়ালীপঈম",(0, 0, 0),(255, 255, 255))
#writing = font.render()
screen.fill((255, 255, 255)) # Clear the background
screen.blit(empty, (10, 10)) # Blit the text surface on the background
pygame.display.flip()  # Refresh the display

input() # Wait for input before quitting
