import pygame
import numpy as np
from PIL import Image
from network import Network

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 560  # 280 for drawing, 280 for prediction
screen_height = 280
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Draw a digit")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
FONT_COLOR = (0, 0, 0)

# Set up fonts
pygame.font.init()
font = pygame.font.SysFont("Unispace", 24)

# Fill the screen with white (left for drawing, right for predictions)
screen.fill(WHITE)
pygame.draw.line(screen, GRAY, (280, 0), (280, 280), 2)  # Dividing line

# Drawing variables
drawing = False
brush_size = 15
erasing = False

nn = Network([784,16,16,10])
nn.load_model("handwritten_digits_V1.0")

# Render the percentages on the right side
def render_predictions(predictions):
    screen.fill(WHITE, (280, 0, 280, 280))  # Clear the right side
    pygame.draw.line(screen, GRAY, (280, 0), (280, 280), 2)  # Dividing line

    labels = [f"{i}: {np.round(pred[0],3)}%" for i, pred in enumerate(predictions)]
    y_offset = 20
    for label in labels:
        text_surface = font.render(label, True, FONT_COLOR)
        screen.blit(text_surface, (300, y_offset))
        y_offset += 25

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and pygame.mouse.get_pos()[0] < 280:
            if event.button == 1:
                drawing = True
            elif event.button == 3:
                erasing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
            elif event.button == 3:
                erasing = False

        # Drawing logic
        if drawing and pygame.mouse.get_pos()[0] < 280:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, BLACK, mouse_pos, 5)
            # Convert the drawing to grayscale and resize to 28x28 for MNIST
            pygame.image.save(screen.subsurface((0, 0, 280, 280)), "digit_drawing.png")
            image = Image.open("digit_drawing.png").convert('L').resize((28, 28))
            image_array = np.array(image) / 255.0  # Normalize the pixel values
            image_array = image_array.reshape(784,1)
            image_array = 1 - image_array  # Invert colors for MNIST (white background, black digit)
            predictions = nn.feed_forward(image_array)

            render_predictions(predictions)

        if erasing and pygame.mouse.get_pos()[0] < 280:
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(screen, WHITE, mouse_pos, 20)
            # Convert the drawing to grayscale and resize to 28x28 for MNIST
            pygame.image.save(screen.subsurface((0, 0, 280, 280)), "digit_drawing.png")
            image = Image.open("digit_drawing.png").convert('L').resize((28, 28))
            image_array = np.array(image) / 255.0  # Normalize the pixel values
            image_array = image_array.reshape(784,1)
            image_array = 1 - image_array  # Invert colors for MNIST (white background, black digit)
            predictions = nn.feed_forward(image_array)

            render_predictions(predictions)

    # Update the display
    pygame.display.update()


# Quit Pygame
pygame.quit()
