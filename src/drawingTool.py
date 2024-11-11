import pygame
import numpy as np
from network import Network
from Button import *
import os

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
NAVY = (56, 75, 112)
GREEN = (143, 209, 79)

# Set up fonts
pygame.font.init()
font = pygame.font.SysFont("courier", 30,bold=True)
medium_size_font = pygame.font.SysFont("courier", 23,bold=True)

# Drawing variables
drawing = False
erasing = False
cell_size = 20
brush_size = 2
canvas_matrix = np.zeros((28,28))

# Screen configuration
screen_width = 28 * cell_size * 2
screen_height = 28 * cell_size + 50
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw a digit")

# Create an instance of the network and load the trained model
model = Network([784, 20, 20, 10])
# Get the current directory of the script being run
current_dir = os.path.dirname(os.path.realpath(__file__))
# Build the path relative to the script location
model_path = os.path.join(current_dir, "../models/handwritten_digits_V1.0")
model.load(model_path)


clear_canvas_button = Button("Clear",(screen_width//2 + 100,screen_height-40),5,font)
settings_button = Button("Settings",(screen_width//2 + 250,screen_height-40),5,font)

draw_instructions = medium_size_font.render("Left Click : draw | Right Click : Erase",True,WHITE)

def draw_predictions(surface, canvas_matrix, start_x, start_y, font):
    # Normalize the canvas matrix before feeding it to the network
    canvas_matrix_normalized = canvas_matrix / 255.0  # Use 255.0 for normalization
    if np.all(canvas_matrix == 0):
        predictions = np.zeros((10))
        surface.blit(font.render("Draw a digit",True,(250,0,0)),(screen_width//7,screen_height//2))
    else:    
        predictions = model.feed_forward(canvas_matrix_normalized.reshape(784, 1))

    predictions = predictions.flatten()
    labels = np.array(["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"])

    sorted_indices = np.argsort(predictions)[::-1]

    predictions = predictions[sorted_indices]
    labels = labels[sorted_indices]

    color = WHITE
    
    for i in range(len(predictions)):

        color = GREEN if i == 0 else WHITE
        label_surface = font.render(f"{labels[i]} : ", True, color)
        prediction_surface = font.render(f"{np.round(predictions[i], 3)}", True, color)
        surface.blit(label_surface, (start_x, start_y))
        surface.blit(prediction_surface, (start_x + label_surface.get_width(), start_y))
        start_y += label_surface.get_height()

def draw_brush(x, y, size, color):
    """ Draw a brush stroke on the surface """
    for i in range(y - size, y + size + 1):
        for j in range(x - size, x + size + 1):
            if 0 <= i < 28 and 0 <= j < 28:
                distance = np.sqrt((j - x) ** 2 + (i - y) ** 2)
                if distance < size:
                    alpha = (size - distance) / size
                    if color == 255:  # Drawing (white)
                        canvas_matrix[i, j] = max(canvas_matrix[i, j], alpha * color)  # Adjust intensity
                    elif color == 0:  # Erasing (black)
                        canvas_matrix[i, j] = 0  # Erase

def draw_canvas(surface):
    """ Draw the canvas with brush strokes """
    for i in range(len(canvas_matrix)):
        for j in range(len(canvas_matrix[i])):
            color_value = int(canvas_matrix[i, j])
            pygame.draw.rect(surface, (color_value, color_value, color_value), (j * cell_size, i * cell_size, cell_size, cell_size))

# Main loop
running = True
while running:
    # Fill the background
    screen.fill(NAVY)
    # Divide the screen into two parts through a vertical line
    pygame.draw.line(screen, BLACK, (screen_width // 2, 0), (screen_width // 2, screen_height), 1)
    
    # Draw the canvas
    draw_canvas(screen)
    #draw the predictions
    draw_predictions(screen, canvas_matrix, screen_width // 2 + 20, 20, font)
    #draw instructions
    screen.blit(draw_instructions,(5,screen_height-draw_instructions.get_height()-10))

    #display buttons and handle clicking
    clear_canvas_button.draw(screen)
    clear_canvas_button.update()
    if clear_canvas_button.pressed:
        canvas_matrix = np.zeros((28,28))

    settings_button.draw(screen)
    settings_button.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pos()[0] < screen_width // 2:
            if event.button == 1:
                drawing = True
            elif event.button == 3:
                erasing = True
        elif event.type == pygame.MOUSEBUTTONUP and pygame.mouse.get_pos()[0] < screen_width // 2:
            if event.button == 1:
                drawing = False
            elif event.button == 3:
                erasing = False

        # Drawing logic
        if drawing and pygame.mouse.get_pos()[0] < screen_width // 2:
            mouse_pos = pygame.mouse.get_pos()
            x, y = mouse_pos[0] // cell_size, mouse_pos[1] // cell_size
            draw_brush(x, y, size=brush_size, color=255)  # Size of the brush
        elif erasing and pygame.mouse.get_pos()[0] < screen_width // 2:
            mouse_pos = pygame.mouse.get_pos()
            x, y = mouse_pos[0] // cell_size, mouse_pos[1] // cell_size
            draw_brush(x, y, size=brush_size, color=0)  # Erase with black

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
