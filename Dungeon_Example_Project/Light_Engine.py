#!/usr/bin/env python

import copy
import math as meth
import random
from typing import NamedTuple, Optional

import numpy as np
import pygame

pygame.init()

pygame.display.set_caption("Light Render")
display_size_pixels = 400, 240
scale_factor = 3
screen_size_pixels = display_size_pixels[0] * scale_factor, display_size_pixels[1] * scale_factor
screen = pygame.display.set_mode(screen_size_pixels, pygame.DOUBLEBUF)
display = pygame.Surface(display_size_pixels)
clock, fps = pygame.time.Clock(), 1000

tileset = pygame.transform.scale(pygame.image.load("Dungeon_Tileset.png"), (160, 160)).convert_alpha()
tile_size_px = 16


def get_tile(x: int, y: int) -> pygame.surface.Surface:
    surf = tileset.copy()
    surf.set_clip(pygame.Rect(x * tile_size_px, y * tile_size_px, tile_size_px, tile_size_px))
    img = surf.subsurface(surf.get_clip())
    return img.copy()


tile_textures = []

for y in range(10):
    for x in range(10):
        tile_textures.append(get_tile(x, y))


class Light:
    def __init__(
        self,
        radius_px: int,
        color: pygame.Color,
        intensity: float,
        is_point: bool,
        angle_deg: float = 0.0,
        angle_width_deg: float = 360.0,
    ) -> None:
        """
        Docstring for __init__

        :param radius_px: The radius of the light in pixels
        :param color: Color of the light
        :param intensity: Intensity of the light [0.0, 1.0]
        :param is_point: If the light is a point (directional) light source
        :param angle_deg: For a point light, the direction of the light's beam in degrees [0.0, 360.0]
        :param angle_width_deg: For a point light, the width of the lights's bean in degrees [0.0, 360.0]
        """
        self.size_px = radius_px * 2
        self.radius_px = radius_px
        self.render_surface = pygame.Surface((self.size_px, self.size_px))
        self.intensity = intensity
        self.angle = angle_deg
        self.angle_width = angle_width_deg
        self.point = is_point
        self.pixel_shader_surf = self.pixel_shader(
            np.full((self.size_px, self.size_px, 3), (color.r, color.g, color.b), dtype=np.uint8)
        )
        self.render_surface.set_colorkey((0, 0, 0))

    def get_intersection(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if dx == 0:
            return [p2[0], (0 if dy <= 0 else self.size_px)]

        if dy == 0:
            return [(0 if dx <= 0 else self.size_px), p2[1]]

        y_gradient = dy / dx
        y_intercept = p1[1] - (p1[0] * y_gradient)

        y_line = 0 if dx <= 0 else self.size_px
        y_intersection = [y_line, (y_gradient * y_line) + y_intercept]

        if y_intersection[1] >= 0 and y_intersection[1] <= self.size_px:
            return y_intersection

        x_gradient = dx / dy
        x_intercept = p1[0] - (p1[1] * x_gradient)

        x_line = 0 if dy <= 0 else self.size_px
        x_intersection = [(x_gradient * x_line) + x_intercept, x_line]

        if x_intersection[0] >= 0 and x_intersection[0] <= self.size_px:
            return x_intersection

    def fill_shadows(self, render_surface: pygame.surface.Surface, points) -> None:
        render_points = [points[0], points[4], points[1], points[2], points[3]]

        if points[2][0] + points[3][0] not in [1000, 0] and points[2][1] + points[3][1] not in [1000, 0]:
            if abs(points[2][0] - points[3][0]) == self.size_px:  # x opposite

                if self.radius_px < points[2][1]:
                    render_points = [
                        points[0],
                        points[4],
                        points[1],
                        points[2],
                        [0, self.size_px],
                        [self.size_px, self.size_px],
                        points[3],
                    ]

                if self.radius_px > points[2][1]:
                    render_points = [points[0], points[4], points[1], points[2], [self.size_px, 0], [0, 0], points[3]]

            elif abs(points[2][1] - points[3][1]) == self.size_px:  # y opposite

                if self.radius_px < points[2][0]:
                    render_points = [
                        points[0],
                        points[4],
                        points[1],
                        points[2],
                        [self.size_px, self.size_px],
                        [self.size_px, 0],
                        points[3],
                    ]

                if self.radius_px > points[2][0]:
                    render_points = [points[0], points[4], points[1], points[2], [0, self.size_px], [0, 0], points[3]]

            else:
                if points[2][0] != self.size_px and points[2][0] != 0:
                    render_points = [
                        points[0],
                        points[4],
                        points[1],
                        points[2],
                        [points[3][0], points[2][1]],
                        points[3],
                    ]

                else:
                    render_points = [
                        points[0],
                        points[4],
                        points[1],
                        points[2],
                        [points[2][0], points[3][1]],
                        points[3],
                    ]

        pygame.draw.polygon(render_surface, (0, 0, 0), render_points)

    def get_corners(self, points, mx, my):
        corners = [points[0], points[2], points[2]]

        if mx >= points[1][0] and mx <= points[0][0]:  # top / bottom
            if my < points[1][1]:
                corners = [points[0], points[1], points[1]]
            if my > points[0][1]:
                corners = [points[2], points[3], points[3]]

        if my >= points[0][1] and my <= points[2][1]:  # left / right
            if mx < points[1][0]:
                corners = [points[1], points[2], points[2]]
            if mx > points[0][0]:
                corners = [points[0], points[3], points[3]]

        if mx < points[1][0] and my < points[1][1]:  # top left / bottom right
            corners = [points[0], points[2], points[1]]
        elif mx > points[0][0] and my > points[2][1]:  # top left / bottom right
            corners = [points[0], points[2], points[3]]

        if mx > points[0][0] and my < points[1][1]:  # top right / bottom left
            corners = [points[1], points[3], points[0]]
        elif mx < points[1][0] and my > points[2][1]:  # top right / bottom left
            corners = [points[1], points[3], points[2]]

        return corners

    def get_tiles(self, tiles: list[list[int]], x_px: int, y_px: int) -> list[list]:
        points = []

        h = len(tiles)
        w = len(tiles[0])
        for y in range(h):
            for x in range(w):
                if tiles[y][x]:
                    if (
                        x * tile_size_px - x_px >= (-self.radius_px) - tile_size_px
                        and x * tile_size_px - x_px <= self.radius_px
                    ) and (
                        y * tile_size_px - y_px >= (-self.radius_px) - tile_size_px
                        and y * tile_size_px - y_px <= self.radius_px
                    ):
                        points.append(
                            [
                                [x * tile_size_px + tile_size_px, y * tile_size_px],
                                [x * tile_size_px, y * tile_size_px],
                                [x * tile_size_px, y * tile_size_px + tile_size_px],
                                [x * tile_size_px + tile_size_px, y * tile_size_px + tile_size_px],
                            ]
                        )

        return points

    def pixel_shader(self, array: np.typing.NDArray[np.uint8]) -> pygame.surface.Surface:
        final_array = np.array(array)

        for x in range(len(final_array)):

            for y in range(len(final_array[x])):

                # radial -----
                distance = meth.sqrt((x - self.radius_px) ** 2 + (y - self.radius_px) ** 2)

                radial_falloff = (self.radius_px - distance) * (1 / self.radius_px)

                if radial_falloff <= 0:
                    radial_falloff = 0
                # -----

                # angular -----
                point_angle = (180 / meth.pi) * -meth.atan2((self.radius_px - x), (self.radius_px - y)) + 180
                diff_anlge = abs(((self.angle - point_angle) + 180) % 360 - 180)

                angular_falloff = ((self.angle_width / 2) - diff_anlge) * (1 / self.angle_width)

                if angular_falloff <= 0:
                    angular_falloff = 0

                if not self.point:
                    angular_falloff = 1
                # -----

                final_intensity = radial_falloff * angular_falloff * self.intensity
                final_array[x][y] = final_array[x][y] * final_intensity

        return pygame.surfarray.make_surface(final_array)

    def check_cast(self, points, dx, dy):
        render = False

        if self.point:
            for point in points:

                try:
                    color = self.pixel_shader_surf.get_at((int(point[0] - dx), int(point[1] - dy)))
                except:
                    color = (0, 0, 0, 255)

                if color != (0, 0, 0, 255):
                    render = True

        else:
            render = True

        return render

    def add_light(
        self, light_surface: pygame.surface.Surface, shadow_tiles: list[list[int]], x_px: int, y_px: int
    ) -> None:

        self.render_surface.fill((0, 0, 0))
        self.render_surface.blit(self.pixel_shader_surf, (0, 0))

        dx, dy = x_px - self.radius_px, y_px - self.radius_px

        for point in self.get_tiles(shadow_tiles, x_px, y_px):

            if self.check_cast(point, dx, dy):

                corners = self.get_corners(point, x_px, y_px)
                corners = [
                    [corners[0][0] - dx, corners[0][1] - dy],
                    [corners[1][0] - dx, corners[1][1] - dy],
                    [corners[2][0] - dx, corners[2][1] - dy],
                ]
                self.fill_shadows(
                    self.render_surface,
                    [
                        corners[0],
                        corners[1],
                        self.get_intersection([self.radius_px] * 2, corners[1]),
                        self.get_intersection([self.radius_px] * 2, corners[0]),
                        corners[2],
                    ],
                )

        pygame.draw.circle(self.render_surface, (255, 255, 255), (self.radius_px, self.radius_px), 2)

        light_surface.blit(self.render_surface, (dx, dy), special_flags=pygame.BLEND_RGBA_ADD)


class Map:
    def __init__(self) -> None:
        # 1 indicates a wall
        self.tiles = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

        # Same si
        self.shadow_tiles = copy.deepcopy(self.tiles)

        # Index in tile_textures for each tile
        self.texture_map = copy.deepcopy(self.tiles)

        self.generate_tiles()

    def render(self, win: pygame.surface.Surface) -> None:
        """Blit all of the tiles in the map to the provided surface."""
        h = len(self.tiles)
        w = len(self.tiles[0])
        for y in range(h):
            for x in range(w):
                tile_pos = [x * tile_size_px, y * tile_size_px]
                win.blit(tile_textures[self.texture_map[y][x]], tile_pos)

    def generate_tiles(self) -> None:
        """Generate self.shadow_tiles and self.texture_map based on self.tiles."""
        h = len(self.tiles)
        w = len(self.tiles[0])
        max_y = h - 1
        for y in range(h):
            for x in range(w):
                self.texture_map[y][x] = self.select_tile_textures_index(x, y)
                self.shadow_tiles[y][x] = self.tiles[y][x] and (y == max_y or self.tiles[y + 1][x])

    def select_tile_textures_index(self, x: int, y: int) -> int:
        """Select an index in tile_textures at random based on which tiles in the local 3x3 block are walls.
        NOTE: This method hardcodes knowledge about tile positions in Dungeon_Tileset.png."""
        max_x = len(self.tiles[0]) - 1
        max_y = len(self.tiles) - 1
        tile_states = [
            self.tiles[y - 1][x - 1] if x > 0 and y > 0 else 1,
            self.tiles[y - 1][x] if y > 0 else 1,
            self.tiles[y - 1][x + 1] if x < max_x and y > 0 else 1,
            self.tiles[y][x - 1] if x > 0 else 1,
            self.tiles[y][x],
            self.tiles[y][x + 1] if x < max_x else 1,
            self.tiles[y + 1][x - 1] if x > 0 and y < max_y else 1,
            self.tiles[y + 1][x] if y < max_y else 1,
            self.tiles[y + 1][x + 1] if x < max_x and y < max_y else 1,
        ]

        # --------------------------------------

        if tile_states[4]:
            if (not tile_states[8] or not tile_states[5]) and tile_states[3] and tile_states[1] and tile_states[7]:
                return random.choice([0, 10, 20, 30])

            elif (not tile_states[6] or not tile_states[3]) and tile_states[5] and tile_states[1] and tile_states[7]:
                return random.choice([5, 15, 25, 35])

            elif not tile_states[7] and tile_states[1]:
                return random.choice([1, 2, 3, 4])

            elif not tile_states[1] and tile_states[3] and tile_states[5] and tile_states[7]:
                return random.choice([41, 42, 43, 44])

            elif not tile_states[2] and tile_states[1] and tile_states[5]:
                return 40

            elif not tile_states[0] and tile_states[1] and tile_states[3]:
                return 45

            elif not tile_states[3] and not tile_states[0] and not tile_states[1]:
                return random.choice([50, 54])

            elif not tile_states[1] and not tile_states[2] and not tile_states[5]:
                return random.choice([53, 55])

            else:
                return 78

        else:
            return random.choice([6, 7, 8, 9, 16, 17, 18, 19, 26, 27, 28, 29])

    def clicking(self, x_px: int, y_px: int, place_wall: bool) -> None:
        """Add or remove a wall tile based on pixel coordinates"""
        self.tiles[y_px // tile_size_px][x_px // tile_size_px] = 1 if place_wall else 0
        self.generate_tiles()


def add_ambient_light(
    light_surface: pygame.surface.Surface, color: Optional[pygame.Color] = None, intensity: int = 50
) -> None:
    """Add ambient light of the specified color and intensity
    :param intensity: [0, 255]
    """
    if color is None:
        color = pygame.Color("white")
    ambient_light = pygame.Surface(light_surface.get_size()).convert_alpha()
    ambient_light.fill((color.r, color.g, color.b, intensity))
    light_surface.blit(ambient_light)


world = Map()

# size, color, intensity, point, angle=0, angle_width=360
mouse_light = Light(75, pygame.Color(255, 185, 9), 1, False)


class MapLight(NamedTuple):
    light: Light
    x_px: int
    y_px: int


# List of tuples of lights and their positions
map_lights: list[MapLight] = []


def screen_to_display_coords(pos_px: tuple[int, int]) -> tuple[int, int]:
    """Account for display to screen scaling"""
    return pos_px[0] // scale_factor, pos_px[1] // scale_factor


while True:
    display.fill((0, 0, 0))

    world.render(display)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x_px, mouse_y_px = screen_to_display_coords(event.pos)
            if event.button == 1:  # left mouse button
                # Add a wall on a left mouse click
                world.clicking(mouse_x_px, mouse_y_px, place_wall=True)
            elif event.button == 2:  # mousewheel
                # Add a light on a mousewheel click
                map_lights.append(
                    MapLight(
                        Light(75, pygame.Color(255, 185, 9), 1, False),
                        mouse_x_px,
                        mouse_y_px,
                    )
                )
            elif event.button == 3:  # right mouse button
                # Remove a wall on a left mouse click
                world.clicking(mouse_x_px, mouse_y_px, place_wall=False)

    light_surface = pygame.Surface((display.get_size()))
    add_ambient_light(light_surface)

    mouse_x_px, mouse_y_px = screen_to_display_coords(pygame.mouse.get_pos())
    mouse_light.add_light(light_surface, world.shadow_tiles, mouse_x_px, mouse_y_px)

    for map_light in map_lights:
        map_light.light.add_light(light_surface, world.shadow_tiles, map_light.x_px, map_light.y_px)
        display.blit(tile_textures[90], (map_light.x_px - tile_size_px // 2, map_light.y_px - tile_size_px // 2))

    display.blit(light_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    surf = pygame.transform.scale(display, screen_size_pixels)
    screen.blit(surf, (0, 0))

    pygame.display.set_caption(str(clock.get_fps()))
    pygame.display.update()
    clock.tick(fps)
