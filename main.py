import pygame
from game import Game


pygame.init()

clock = pygame.time.Clock()

game = Game(headless=False)


while True:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                game.reset(seed=0)

    if game.running:

        game.handle_input()

        game.update()

    else:

        keys = pygame.key.get_pressed()

        if keys[pygame.K_r]:
            game.reset(seed=0)

    game.draw()

    clock.tick(60)