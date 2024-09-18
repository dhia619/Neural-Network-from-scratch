import pygame

pygame.init()

class Button:
    def __init__(self,text,pos,elevation,font):
        self.text = text
        self.elevation = elevation
        self.text_surface = font.render(self.text,True,(255,255,255))
        self.rect = pygame.Rect(pos[0]-self.text_surface.get_width()//2-8,
                                pos[1]-self.text_surface.get_height()//2-8
                                ,self.text_surface.get_width()+16,self.text_surface.get_height()+13)
        self.on_color = (25,76,160)
        self.off_color = (123, 143, 161)
        self.bottom_rect_color = (207, 185, 151)
        self.bottom_rect = pygame.Rect(self.rect.x,self.rect.y+self.elevation,self.rect.width,self.rect.height)
        self.color = self.off_color
        self.pressed = False
    def draw(self,screen):
        pygame.draw.rect(screen,self.bottom_rect_color,self.bottom_rect,border_radius=8)
        pygame.draw.rect(screen,self.color,self.rect,border_radius=8)
        screen.blit(self.text_surface,(self.rect.centerx-self.text_surface.get_width()//2,self.rect.centery-self.text_surface.get_height()//2))
    def update(self):
        mouse = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            self.color = self.on_color
            if mouse[0]:
                self.pressed = True
                self.rect.y = self.bottom_rect.y
            else:
                self.rect.y = self.bottom_rect.y-self.elevation
                self.pressed = False
        else:
            self.rect.y = self.bottom_rect.y-self.elevation
            self.color = self.off_color

class TextEntry:
    def __init__(self,text,pos,width,height,font):
        self.rect = pygame.Rect(pos[0],pos[1],width,height)
        self.color = (244, 230, 231)
        self.border_color = (10, 19, 19)
        self.focus = False
        self.text = ""
        self.font = font
        self.placeholder = font.render("Aa",True,(115,115,115))
    def draw(self,screen,mouse_pos):
        if len(self.text)>=14 and not(self.focus):  
            self.text_surface = self.font.render(self.text[:15],True,(0,0,0))
        else:
            self.text_surface = self.font.render(self.text,True,(0,0,0)) 
        pygame.draw.rect(screen,self.color,self.rect,border_radius=11)
        pygame.draw.rect(screen,self.border_color,self.rect,border_radius=11,width=2)
        screen.blit(self.text_surface,(self.rect.left+10,self.rect.centery-self.text_surface.get_height()//2))
        if self.rect.collidepoint(mouse_pos):
            self.border_color = (71, 60, 51)
        else:
            self.border_color = (10, 19, 19)
        if self.text.strip() == "":
           screen.blit(self.placeholder,(self.rect.x+5,self.rect.y+8))
        if self.focus:
            self.border_color = (71, 60, 51)
            if self.rect.width <= 310:
                self.rect.width += 4
                self.rect.x -= 4
        else:
            if self.rect.width >= 200:
                self.rect.width -= 4
                self.rect.x += 4
    def enable(self):
        if self.rect.collidepoint(mouse_pos):
                self.focus = True
        else:
            self.focus = False