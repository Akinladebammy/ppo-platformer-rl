import pygame


class Player(pygame.sprite.Sprite):

    def __init__(self, position):
        super().__init__()

        self.image = pygame.Surface((32, 48))
        self.image.fill("blue")

        self.rect = self.image.get_rect(topleft=position)

        self.direction = pygame.math.Vector2(0, 0)

        self.speed = 5
        self.gravity = 0.8
        self.jump_speed = -14

        self.on_ground = False

    def move_left(self):
        self.direction.x = -1

    def move_right(self):
        self.direction.x = 1

    def stop(self):
        self.direction.x = 0

    def jump(self):
        if self.on_ground:
            self.direction.y = self.jump_speed
            self.on_ground = False

    def apply_gravity(self):
        self.direction.y += self.gravity


class Platform(pygame.sprite.Sprite):

    def __init__(self, x, y, width, height):
        super().__init__()

        self.image = pygame.Surface((width, height))
        self.image.fill("forestgreen")

        self.rect = self.image.get_rect(topleft=(x, y))


class Enemy(pygame.sprite.Sprite):

    def __init__(self, position):
        super().__init__()

        self.image = pygame.Surface((32, 32))
        self.image.fill("red")

        self.rect = self.image.get_rect(topleft=position)

        self.speed = 2
        self.direction = -1

        self.start_x = position[0]
        self.patrol_distance = 120

    def update(self):
        self.rect.x += self.speed * self.direction

        if self.rect.x <= self.start_x - self.patrol_distance:
            self.direction = 1

        if self.rect.x >= self.start_x:
            self.direction = -1


class Hazard(pygame.sprite.Sprite):

    def __init__(self, position):
        super().__init__()

        self.image = pygame.Surface((32, 32))
        self.image.fill("orange")

        self.rect = self.image.get_rect(topleft=position)


class Goal(pygame.sprite.Sprite):

    def __init__(self, position):
        super().__init__()

        self.image = pygame.Surface((20, 100))
        self.image.fill("yellow")

        self.rect = self.image.get_rect(bottomleft=position)