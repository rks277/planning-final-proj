import cv2
import numpy as np
import copy
import math
import time
from PIL import Image as PILImage
from IPython.display import Image, display

# ## Generic Image Utils

def load_image(filename, scale=1.0):
  img = cv2.imread(filename)
  if img is None:
    raise FileNotFoundError(f"Could not load image: {filename}")
  return scale_image(img, scale)

def scale_image(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def show_image(img, scale=1.0):
  if not scale == 1.0:
    display(PILImage.fromarray(cv2.cvtColor(scale_image(img, scale), cv2.COLOR_BGR2RGB)))
  else:
    display(PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

def world_height_and_width(img):
  shape = img.shape
  (pixel_height, pixel_width) = (shape[0], shape[1])
  world_height = 1.0
  world_width = pixel_width / pixel_height
  return (world_height, world_width)

def image_space_to_world_space(img, y, x, clamp=False):
  shape = img.shape
  (pixel_height, pixel_width) = (shape[0], shape[1])
  world_height = 1.0
  world_width = pixel_width / pixel_height

  world_space_y = y / pixel_height
  if clamp:
    world_space_y = min(1.0, max(world_space_y, 0.0))

  world_space_x = (x / pixel_width) * world_width
  if clamp:
    world_space_x = min(1.0, max(world_space_x, 0.0))
  return (world_space_y, world_space_x)

def world_space_to_image_space(img, y, x, clamp=False):
  shape = img.shape
  (pixel_height, pixel_width) = (shape[0], shape[1])
  world_height = 1.0
  world_width = pixel_width / pixel_height

  image_space_y = round(pixel_height * y)
  if clamp:
    image_space_y = max(0, min(image_space_y, pixel_height-1))
  image_space_x = round((pixel_width * x) / world_width)
  if clamp:
    image_space_x = max(0, min(image_space_x, pixel_width-1))

  return (image_space_y, image_space_x)

def world_space_to_image_space_scalar(img, scalar):
  shape = img.shape
  (pixel_height, pixel_width) = (shape[0], shape[1])
  return round(pixel_height * scalar)

def is_point_in_collision(img, y_world, x_world):
  return False


# ## Drawing & GIF Utils

def output_gif_from_cv2_imgs(imgs, output_name, fps=30, num_seconds_at_end=1):
  if len(imgs) == 0:
    print('could not output {}.gif because the image stack was empty.'.format(output_name))
    return

  stride = 1
  if fps > 30:
    stride = round(fps / 30)
  image_imgs = []
  for img in imgs[0::stride]:
    ii = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_imgs.append(PILImage.fromarray(ii))
  if fps > 30:
    ii = cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2RGB)
    image_imgs.append(PILImage.fromarray(ii))
  duration = (1.0/(min(30, fps)))*1000.0
  duration_sum = 0.0
  while duration_sum < num_seconds_at_end*1000.0:
    ii = cv2.cvtColor(imgs[-1], cv2.COLOR_BGR2RGB)
    image_imgs.append(PILImage.fromarray(ii))
    duration_sum += duration

  frame_one = image_imgs[0]
  frame_one.save('{}.gif'.format(output_name), format="GIF", append_images=image_imgs, save_all=True, duration=duration, loop=0)
  print('{}.gif has been created.'.format(output_name))


# ## Continuous Planner Util

class ContinuousPlannerUtil:
  def __init__(self, bg_filename, density_filename, scale=1.0):
    self.raw_img = load_image(bg_filename, scale=scale)
    self.density_img = load_image(density_filename, scale=scale)

    self.density_gray = cv2.cvtColor(self.density_img, cv2.COLOR_BGR2GRAY)
    self.density_map = self.density_gray.astype(np.float32) / 255.0

    self.draw_img = self.raw_img.copy()

    heatmap = cv2.applyColorMap(self.density_gray, cv2.COLORMAP_JET)
    self.draw_img = cv2.addWeighted(self.draw_img, 0.6, heatmap, 0.4, 0)

    (self.height, self.width) = world_height_and_width(self.raw_img)
    self.gif_image_stack = []

  def get_density(self, state):
    [y, x] = world_space_to_image_space(self.raw_img, state[0], state[1], clamp=True)
    return self.density_map[y, x]

  def sample_state(self):
    return np.array([np.random.uniform(high=self.height), np.random.uniform(high=self.width)])

  def is_state_in_collision(self, state):
    if state[0] < 0.0 or state[0] > self.height or state[1] < 0.0 or state[1] > self.width:
      return True
    return False

  def is_straight_line_connection_feasible(self, state_1, state_2, step_size = 0.01):
    state_1_arr = np.array(state_1)
    state_2_arr = np.array(state_2)
    dis = np.linalg.norm(state_1_arr - state_2_arr)
    aa = np.linspace(0.0, 1.0, math.ceil(dis / step_size))
    if len(aa) < 2: aa = [0.0, 1.0]
    for a in aa:
      s = (1.0 - a)*state_1_arr + a*state_2_arr
      if self.is_state_in_collision(s):
        return False
    return True

  def draw_path(self, path, color=(183, 21, 212), thickness_world=0.015, alpha=0.6):
      overlay = self.draw_img.copy()
      thick = max(world_space_to_image_space_scalar(self.draw_img, thickness_world), 1)

      for i in range(len(path) - 1):
          p1 = path[i]
          p2 = path[i+1]
          (y1, x1) = world_space_to_image_space(self.draw_img, p1[0], p1[1])
          (y2, x2) = world_space_to_image_space(self.draw_img, p2[0], p2[1])
          cv2.line(overlay, (x1, y1), (x2, y2), color, thick)

      self.draw_img = cv2.addWeighted(overlay, alpha, self.draw_img, 1 - alpha, 0)

  def draw_tree(self, vertex_list, edge_list, clear_draw_img=False):
    if clear_draw_img:
      self.draw_img = cv2.addWeighted(self.raw_img.copy(), 0.6, cv2.applyColorMap(self.density_gray, cv2.COLORMAP_JET), 0.4, 0)

    overlay = self.draw_img.copy()
    line_thickness = 1
    r_world = 0.006
    r = world_space_to_image_space_scalar(self.draw_img, r_world)
    alpha = 0.7

    for edge in edge_list.edges:
      v1 = edge.vertex_1
      v2 = edge.vertex_2
      (y1, x1) = world_space_to_image_space(self.draw_img, v1.state[0], v1.state[1])
      (y2, x2) = world_space_to_image_space(self.draw_img, v2.state[0], v2.state[1])
      cv2.line(overlay, (x1, y1), (x2, y2), edge.bgr_color, line_thickness)

    self.draw_img = cv2.addWeighted(overlay, alpha, self.draw_img, 1 - alpha, 0)

    for v in vertex_list.vertices:
        (y, x) = world_space_to_image_space(self.draw_img, v.state[0], v.state[1])
        cv2.circle(self.draw_img, (x,y), r, v.bgr_color, -1)

  def show_image(self, scale=1.0):
    show_image(self.draw_img, scale=scale)

  def save_image_to_gif_image_stack(self):
    self.gif_image_stack.append(self.draw_img.copy())

  def output_gif_animation(self, output_name, fps=30, num_seconds_at_end=1):
    output_gif_from_cv2_imgs(self.gif_image_stack, output_name, fps, num_seconds_at_end)


# ## Graph Structures with Accumulated State

class Vertex:
  def __init__(self, state, bgr_color=(255,200,30), parent=None, dist_acc=0.0, plastic_acc=0.0, cost=0.0):
    self.state = np.array(state)
    self.outgoing_edges = []
    self.incoming_edges = []
    self.bgr_color = bgr_color
    self.parent = parent

    # Accumulated values from start to this vertex
    self.dist_acc = dist_acc
    self.plastic_acc = plastic_acc

    # Weighted cost for sorting/optimization
    self.cost = cost

class Edge:
  def __init__(self, vertex_1, vertex_2, length, plastic_collected, bgr_color=(30,30,30)):
    self.vertex_1 = vertex_1
    self.vertex_2 = vertex_2
    self.vertex_1.outgoing_edges.append(self)
    self.vertex_2.incoming_edges.append(self)

    self.length = length
    self.plastic_collected = plastic_collected

    self.bgr_color = bgr_color

class VertexList:
  def __init__(self):
    self.vertices = []

  def add_vertex(self, vertex):
    self.vertices.append(vertex)
    return vertex

  def get_closest_vertex_in_graph(self, state):
    closest_vertex = None
    closest_dis = float('inf')
    state_arr = np.array(state)
    for vertex in self.vertices:
      dis = np.linalg.norm(vertex.state - state_arr)
      if dis < closest_dis:
        closest_vertex = vertex
        closest_dis = dis
    return (closest_vertex, closest_dis)

  def get_vertices_within_distance_r(self, state, r):
    output = []
    state_arr = np.array(state)
    for vertex in self.vertices:
      dis = np.linalg.norm(vertex.state - state_arr)
      if dis < r:
        output.append((vertex, dis))
    return output

class EdgeList:
  def __init__(self):
    self.edges = []

  def add_edge(self, edge):
    self.edges.append(edge)
    return edge

  def delete_edge(self, v1, v2):
      to_remove = []
      for e in self.edges:
          if e.vertex_1 == v1 and e.vertex_2 == v2:
              to_remove.append(e)
      for e in to_remove:
          if e in v1.outgoing_edges: v1.outgoing_edges.remove(e)
          if e in v2.incoming_edges: v2.incoming_edges.remove(e)
          self.edges.remove(e)
