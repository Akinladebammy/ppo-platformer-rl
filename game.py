import pygame

from sprites import Player, Platform, Enemy, Hazard, Goal


class Game:

    WIDTH = 1000
    HEIGHT = 600

    def __init__(self, headless=False, seed=0):

        pygame.init()

        self.font = pygame.font.Font(None, 48)

        self.headless = headless
        self.seed = seed

        if not self.headless:
            self.screen = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT)
            )
            pygame.display.set_caption("RL Platformer")
        else:
            self.screen = pygame.Surface(
                (self.WIDTH, self.HEIGHT)
            )

        self.reset(seed)


    def reset(self, seed=0):

        self.seed = seed

        self.player = pygame.sprite.GroupSingle()
        self.platforms = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.hazards = pygame.sprite.Group()
        self.goal = pygame.sprite.GroupSingle()

        self.player.add(
            Player((50, 450))
        )

        self.platforms.add(
            Platform(0, 550, 300, 50),
            Platform(380, 550, 250, 50),
            Platform(700, 550, 300, 50),

            Platform(180, 450, 120, 20),
            Platform(470, 470, 120, 20),
        )

        self.enemies.add(
            Enemy((550, 518))
        )

        self.hazards.add(
            Hazard((760, 518))
        )

        self.goal.add(
            Goal((940, 550))
        )

        self.running = True
        self.won = False

        return self.get_state()


    def get_state(self):

        player = self.player.sprite

        return {
            "player_x": player.rect.x,
            "player_y": player.rect.y,
            "velocity_x": player.direction.x,
            "velocity_y": player.direction.y,
            "on_ground": player.on_ground,
            "won": self.won,
        }


    def handle_input(self):

        keys = pygame.key.get_pressed()

        player = self.player.sprite

        player.stop()

        if keys[pygame.K_LEFT]:
            player.move_left()

        if keys[pygame.K_RIGHT]:
            player.move_right()

        if keys[pygame.K_SPACE]:
            player.jump()


    def take_action(self, action):

        player = self.player.sprite

        player.stop()

        if action == 1:
            player.move_left()

        elif action == 2:
            player.move_right()

        elif action == 3:
            player.jump()

        elif action == 4:
            player.move_left()
            player.jump()

        elif action == 5:
            player.move_right()
            player.jump()


    def horizontal_collision(self):

        player = self.player.sprite

        player.rect.x += int(
            player.direction.x * player.speed
        )

        for platform in self.platforms:

            if player.rect.colliderect(platform.rect):

                if player.direction.x > 0:
                    player.rect.right = platform.rect.left

                elif player.direction.x < 0:
                    player.rect.left = platform.rect.right


    def vertical_collision(self):

        player = self.player.sprite

        player.apply_gravity()


        player.rect.y += int(player.direction.y)

        player.on_ground = False

        for platform in self.platforms:

            if player.rect.colliderect(platform.rect):

                if player.direction.y > 0:
                    player.rect.bottom = platform.rect.top
                    player.direction.y = 0
                    player.on_ground = True

                elif player.direction.y < 0:
                    player.rect.top = platform.rect.bottom
                    player.direction.y = 0


    def check_death(self):

        player = self.player.sprite

        if player.rect.top > self.HEIGHT:
            self.running = False

        if pygame.sprite.spritecollide(
            player,
            self.enemies,
            False
        ):
            self.running = False

        if pygame.sprite.spritecollide(
            player,
            self.hazards,
            False
        ):
            self.running = False


    def check_goal(self):

        player = self.player.sprite

        if pygame.sprite.spritecollide(
            player,
            self.goal,
            False
        ):
            self.won = True
            self.running = False


    def update(self):

        self.horizontal_collision()
        self.vertical_collision()

        self.enemies.update()

        self.check_death()
        self.check_goal()


    def draw(self):

        self.screen.fill("skyblue")

        self.platforms.draw(self.screen)
        self.enemies.draw(self.screen)
        self.hazards.draw(self.screen)
        self.goal.draw(self.screen)
        self.player.draw(self.screen)

        if not self.running:

            if self.won:
                message = "You Win! Press R to restart"
            else:
                message = "Game Over! Press R to restart"

            text = self.font.render(
                message,
                True,
                "black"
            )

            text_rect = text.get_rect(
                center=(
                    self.WIDTH // 2,
                    self.HEIGHT // 2
                )
            )

            self.screen.blit(
                text,
                text_rect
            )

        if not self.headless:
            pygame.display.update()