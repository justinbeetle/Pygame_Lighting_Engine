#!/usr/bin/env python

import copy
import math
import random
from typing import NamedTuple, Optional, Union

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


class Point(NamedTuple):
    x: int
    y: int


class FloatPoint(NamedTuple):
    x: float
    y: float


class CornerCoords(NamedTuple):  # This effective duplicates a rect
    top_right: Point
    top_left: Point
    bottom_left: Point
    bottom_right: Point


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
        self.color = color
        self.intensity = intensity
        self.angle = angle_deg
        self.angle_width_deg = angle_width_deg
        self.is_point = is_point
        self.pixel_shader_surf = self.pixel_shader()
        self.render_surface.set_colorkey((0, 0, 0))

    def get_intersection(self, p1: Point, p2: Point) -> FloatPoint:
        dx = p2.x - p1.x
        dy = p2.y - p1.y

        if dx == 0:
            return FloatPoint(p2.x, (0 if dy <= 0 else self.size_px))

        if dy == 0:
            return FloatPoint((0 if dx <= 0 else self.size_px), p2.y)

        y_gradient = dy / dx
        y_intercept = p1.y - (p1.x * y_gradient)

        y_line = 0 if dx <= 0 else self.size_px
        y_intersection = FloatPoint(y_line, (y_gradient * y_line) + y_intercept)

        if y_intersection.y >= 0 and y_intersection.y <= self.size_px:
            return y_intersection

        x_gradient = dx / dy
        x_intercept = p1.x - (p1.y * x_gradient)

        x_line = 0 if dy <= 0 else self.size_px
        x_intersection = FloatPoint((x_gradient * x_line) + x_intercept, x_line)

        if x_intersection.x >= 0 and x_intersection.x <= self.size_px:
            return x_intersection

    def fill_shadows(self, render_surface: pygame.surface.Surface, point0: Point, point1: Point, point2: FloatPoint, point3: FloatPoint, point4: Point) -> None:
        render_points: list[Union[FloatPoint, Point]] = [point0, point4, point1, point2, point3]

        # TODO: Where does [1000, 0] come from?
        if point2.x + point3.x not in [1000, 0] and point2.y + point3.y not in [1000, 0]:
            if abs(point2.x - point3.x) == self.size_px:  # x opposite

                if self.radius_px < point2.y:
                    render_points = [
                        point0,
                        point4,
                        point1,
                        point2,
                        Point(0, self.size_px),
                        Point(self.size_px, self.size_px),
                        point3,
                    ]

                if self.radius_px > point2.y:
                    render_points = [point0, point4, point1, point2, Point(self.size_px, 0), Point(0, 0), point3]

            elif abs(point2.y - point3.y) == self.size_px:  # y opposite

                if self.radius_px < point2.x:
                    render_points = [
                        point0,
                        point4,
                        point1,
                        point2,
                        Point(self.size_px, self.size_px),
                        Point(self.size_px, 0),
                        point3,
                    ]

                if self.radius_px > point2.x:
                    render_points = [point0, point4, point1, point2, Point(0, self.size_px), Point(0, 0), point3]

            else:
                if point2.x != self.size_px and point2.x != 0:
                    render_points = [
                        point0,
                        point4,
                        point1,
                        point2,
                        FloatPoint(point3.x, point2.y),
                        point3,
                    ]

                else:
                    render_points = [
                        point0,
                        point4,
                        point1,
                        point2,
                        FloatPoint(point2.x, point3.y),
                        point3,
                    ]

        pygame.draw.polygon(render_surface, (0, 0, 0), render_points)

    def get_corners(self, points: CornerCoords, mx: int, my: int) -> tuple[Point, Point, Point]:
        # What are the meaning of the returned 3 returned points and their ordering?

        if mx >= points.top_left.x and mx <= points.top_right.x:  # top / bottom
            if my < points.top_left.y:
                return points.top_right, points.top_left, points.top_left
            if my > points.top_right.y:
                return points.bottom_left, points.bottom_right, points.bottom_right

        if my >= points.top_right.y and my <= points.bottom_left.y:  # left / right
            if mx < points.top_left.x:
                return points.top_left, points.bottom_left, points.bottom_left
            if mx > points.top_right.x:
                return points.top_right, points.bottom_right, points.bottom_right

        if mx < points.top_left.x and my < points.top_left.y:  # top left / bottom right
            return points.top_right, points.bottom_left, points.top_left
        elif mx > points.top_right.x and my > points.bottom_left.y:  # top left / bottom right
            return points.top_right, points.bottom_left, points.bottom_right

        if mx > points.top_right.x and my < points.top_left.y:  # top right / bottom left
            return points.top_left, points.bottom_right, points.top_right
        elif mx < points.top_left.x and my > points.bottom_left.y:  # top right / bottom left
            return points.top_left, points.bottom_right, points.bottom_left

        return points.top_right, points.bottom_left, points.bottom_left

    def get_tiles(self, tiles: list[list[int]], x_px: int, y_px: int) -> list[CornerCoords]: # list[pygame.Rect]:
        """From the shadow tiles, get corner coordinates of each shadow tile within in the
        range of our light."""
        # TODO: Convert to return a list of rects
        # shadow_tile_rects = []
        points = []

        light_rect = pygame.Rect(x_px - self.radius_px, y_px - self.radius_px, self.size_px, self.size_px)
        h = len(tiles)
        w = len(tiles[0])
        for y in range(h):
            for x in range(w):
                if tiles[y][x]:
                    tile_rect = pygame.Rect(x*tile_size_px, y*tile_size_px, tile_size_px, tile_size_px)
                    if light_rect.colliderect(tile_rect):
                        #shadow_tile_rects.append(tile_rect)
                        points.append(
                            # Could just as well append tile_rect here!
                            CornerCoords(
                                Point(x * tile_size_px + tile_size_px, y * tile_size_px),
                                Point(x * tile_size_px, y * tile_size_px),
                                Point(x * tile_size_px, y * tile_size_px + tile_size_px),
                                Point(x * tile_size_px + tile_size_px, y * tile_size_px + tile_size_px),
                            )
                        )

        return points
        #return shadow_tile_rects

    def pixel_shader(self) -> pygame.surface.Surface:
        final_array = np.full(
            (self.size_px, self.size_px, 3), (self.color.r, self.color.g, self.color.b), dtype=np.float64
        )

        # Grid -----
        x, y = np.meshgrid(np.arange(self.size_px), np.arange(self.size_px))
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        # -----

        # Radial -----
        distance = np.sqrt((x - self.radius_px) ** 2 + (y - self.radius_px) ** 2)
        radial_falloff = (self.radius_px - distance) * (1 / self.radius_px)
        radial_falloff[radial_falloff <= 0] = 0
        # -----

        # Angular -----
        if self.is_point:
            point_angle = (180 / np.pi) * -np.arctan2((self.radius_px - x), (self.radius_px - y)) + 180
            diff_angle = np.abs(((self.angle - point_angle) + 180) % 360 - 180)
            angular_falloff = ((self.angle_width_deg / 2) - diff_angle) * (1 / self.angle_width_deg)
            angular_falloff[angular_falloff <= 0] = 0
        else:
            angular_falloff = 1
        # -----

        final_intensity = radial_falloff * angular_falloff * self.intensity
        final_array *= final_intensity[..., np.newaxis]

        return pygame.surfarray.make_surface(final_array.astype(np.uint8))

    def check_cast(self, points: CornerCoords, dx: int, dy: int) -> bool:
        render = False

        if self.is_point:
            for point in points:

                try:
                    color = self.pixel_shader_surf.get_at((point.x - dx, point.y - dy))
                except:
                    color = pygame.Color(0, 0, 0, 255)

                render = color.rgba != (0, 0, 0, 255)

        else:
            render = True

        return render

    def add_light(
        self, light_surface: pygame.surface.Surface, shadow_tiles: list[list[int]], x_px: int, y_px: int
    ) -> None:

        self.render_surface.fill((0, 0, 0))
        self.render_surface.blit(self.pixel_shader_surf, (0, 0))
        radius_pt = Point(self.radius_px, self.radius_px)

        dx, dy = x_px - self.radius_px, y_px - self.radius_px

        for point in self.get_tiles(shadow_tiles, x_px, y_px):

            if self.check_cast(point, dx, dy):

                corners = self.get_corners(point, x_px, y_px)
                corners = (
                    Point(corners[0].x - dx, corners[0].y - dy),
                    Point(corners[1].x - dx, corners[1].y - dy),
                    Point(corners[2].x - dx, corners[2].y - dy),
                )
                self.fill_shadows(
                    self.render_surface,
                    corners[0],
                    corners[1],
                    self.get_intersection(radius_pt, corners[1]),
                    self.get_intersection(radius_pt, corners[0]),
                    corners[2],
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


class MapLight(NamedTuple):
    light: Light
    x_px: int
    y_px: int


# List of tuples of lights and their positions
map_lights: list[MapLight] = []


def screen_to_display_coords(pos_px: tuple[int, int]) -> tuple[int, int]:
    """Account for display to screen scaling"""
    return pos_px[0] // scale_factor, pos_px[1] // scale_factor


last_mouse_x_px: Optional[int] = None
last_mouse_y_px: Optional[int] = None
last_diff_mouse_x_px: Optional[int] = None
last_diff_mouse_y_px: Optional[int] = None
directional_mouse_light: Optional[Light] = None
non_directional_mouse_light = Light(75, pygame.Color(255, 185, 9), 1, False)
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
    non_directional_mouse_light.add_light(light_surface, world.shadow_tiles, mouse_x_px, mouse_y_px)
    if False:
        if last_mouse_x_px is None or last_mouse_y_px is None:
            last_mouse_x_px, last_mouse_y_px = mouse_x_px, mouse_y_px
        if (mouse_x_px, mouse_y_px) != (last_mouse_x_px, last_mouse_y_px):
            last_diff_mouse_x_px, last_diff_mouse_y_px = last_mouse_x_px, last_mouse_y_px
        if last_diff_mouse_x_px is not None and last_diff_mouse_y_px is not None:
            angle_deg = math.atan2(last_diff_mouse_y_px - mouse_y_px, mouse_x_px - last_diff_mouse_x_px) * 180 / np.pi
            directional_mouse_light = Light(150, pygame.Color(255, 255, 255), 1, True, angle_deg, 10)
        if directional_mouse_light is not None:
            directional_mouse_light.add_light(light_surface, world.shadow_tiles, mouse_x_px, mouse_y_px)
        last_mouse_x_px, last_mouse_y_px = mouse_x_px, mouse_y_px

    for map_light in map_lights:
        map_light.light.add_light(light_surface, world.shadow_tiles, map_light.x_px, map_light.y_px)
        display.blit(tile_textures[90], (map_light.x_px - tile_size_px // 2, map_light.y_px - tile_size_px // 2))

    display.blit(light_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    surf = pygame.transform.scale(display, screen_size_pixels)
    screen.blit(surf, (0, 0))

    pygame.display.set_caption(str(clock.get_fps()))
    pygame.display.update()
    clock.tick(fps)
