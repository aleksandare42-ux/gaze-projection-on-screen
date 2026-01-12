import pygame
pygame.init()

W, H = 600, 400
win = pygame.display.set_mode((W, H))
pygame.display.set_caption("Мини-Понг")

x, y = W // 2, H // 2
vx, vy = 4, 4
paddle_x = W // 2 - 50
clock = pygame.time.Clock()

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle_x > 0:
        paddle_x -= 6
    if keys[pygame.K_RIGHT] and paddle_x < W - 100:
        paddle_x += 6

    x += vx
    y += vy

    if x <= 10 or x >= W - 10:
        vx = -vx
    if y <= 10:
        vy = -vy
    if y >= H - 30 and paddle_x < x < paddle_x + 100:
        vy = -vy
    elif y > H:
        running = False

    win.fill((20, 20, 20))
    pygame.draw.circle(win, (0, 255, 120), (x, y), 10)
    pygame.draw.rect(win, (200, 200, 200), (paddle_x, H - 20, 100, 10))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
