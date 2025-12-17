import numpy as np
import math
import planning_lib
from planning_lib import ContinuousPlannerUtil, Vertex, Edge, VertexList, EdgeList

# ## GPGP Experiment Cost Functions

def integrate_path_segment(c, p1, p2, step_size=0.01):
    # Returns (distance, plastic_collected)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    if dist < 1e-6: return 0.0, 0.0

    steps = math.ceil(dist / step_size)
    if steps < 1: steps = 1

    plastic_sum = 0.0

    # Trapezoidal rule approximation or just mid-point sampling
    ts = np.linspace(0.0, 1.0, steps + 1)
    for i in range(len(ts) - 1):
        t_start = ts[i]
        t_end = ts[i+1]
        mid_t = (t_start + t_end) / 2.0

        pos = (1.0 - mid_t) * np.array(p1) + mid_t * np.array(p2)
        density = c.get_density(pos)

        segment_len = (t_end - t_start) * dist
        plastic_sum += density * segment_len

    return dist, plastic_sum

def calculate_total_cost(dist_acc, plastic_acc, alpha, mode="linear"):
    if mode == "linear":
        # Cost = TotalDistance - alpha * TotalPlastic
        return dist_acc - alpha * plastic_acc
    elif mode == "ratio":
        # Cost = TotalDistance / (1 + alpha * TotalPlastic)
        # Using 1 + ... to avoid division by zero if plastic is 0 or very small
        return dist_acc / (1.0 + alpha * plastic_acc)
    else:
        raise ValueError(f"Unknown cost mode: {mode}")

def rrt_star_tunable(c, alpha=1.0, loops=800, output_prefix="rrt_tunable", cost_mode="linear"):
  print(f"Running RRT* with alpha={alpha}, mode={cost_mode}...")
  start_state = [0.5, 0.1]
  goal_state = [0.5, 1.2]

  vertex_list = VertexList()
  edge_list = EdgeList()

  start_vertex = vertex_list.add_vertex(Vertex(start_state, bgr_color=(200, 255, 30), dist_acc=0.0, plastic_acc=0.0, cost=0.0))

  step_size = 0.08
  search_radius = 0.15
  goal_bias = 0.1

  for i in range(loops):
      # 1. Sample
      if np.random.random() < goal_bias:
          sample = np.array(goal_state)
      else:
          sample = c.sample_state()

      # 2. Nearest
      (nearest_v, _) = vertex_list.get_closest_vertex_in_graph(sample)

      # 3. Steer
      direction = sample - nearest_v.state
      length = np.linalg.norm(direction)
      if length < 1e-6: continue

      direction = direction / length
      new_state = nearest_v.state + direction * min(step_size, length)

      if not c.is_straight_line_connection_feasible(nearest_v.state, new_state):
          continue

      # Calculate incremental values for potential new edge
      d_edge, p_edge = integrate_path_segment(c, nearest_v.state, new_state)

      # 4. Choose Parent
      neighbors = vertex_list.get_vertices_within_distance_r(new_state, search_radius)

      # Default best is nearest
      best_parent = nearest_v
      min_total_cost = calculate_total_cost(nearest_v.dist_acc + d_edge, nearest_v.plastic_acc + p_edge, alpha, mode=cost_mode)
      best_d_edge = d_edge
      best_p_edge = p_edge

      for (v, _) in neighbors:
          if not c.is_straight_line_connection_feasible(v.state, new_state): continue

          d_v, p_v = integrate_path_segment(c, v.state, new_state)
          cost_v = calculate_total_cost(v.dist_acc + d_v, v.plastic_acc + p_v, alpha, mode=cost_mode)

          if cost_v < min_total_cost:
              min_total_cost = cost_v
              best_parent = v
              best_d_edge = d_v
              best_p_edge = p_v

      new_vertex = vertex_list.add_vertex(Vertex(
          new_state, 
          parent=best_parent, 
          dist_acc=best_parent.dist_acc + best_d_edge, 
          plastic_acc=best_parent.plastic_acc + best_p_edge,
          cost=min_total_cost
      ))
      edge_list.add_edge(Edge(best_parent, new_vertex, best_d_edge, best_p_edge))

      # 5. Rewire
      for (v, _) in neighbors:
          if v == best_parent: continue
          if not c.is_straight_line_connection_feasible(new_vertex.state, v.state): continue

          d_rewire, p_rewire = integrate_path_segment(c, new_vertex.state, v.state)
          rewire_total_cost = calculate_total_cost(new_vertex.dist_acc + d_rewire, new_vertex.plastic_acc + p_rewire, alpha, mode=cost_mode)

          if rewire_total_cost < v.cost:
              # Check for cycles: v should not be an ancestor of new_vertex
              # new_vertex.parent is best_parent. So we check if v is ancestor of best_parent.
              curr = best_parent
              is_cycle = False
              while curr:
                  if curr == v:
                      is_cycle = True
                      break
                  curr = curr.parent
              
              if is_cycle:
                  continue

              if v.parent:
                  edge_list.delete_edge(v.parent, v)
              v.parent = new_vertex
              v.dist_acc = new_vertex.dist_acc + d_rewire
              v.plastic_acc = new_vertex.plastic_acc + p_rewire
              v.cost = rewire_total_cost
              edge_list.add_edge(Edge(new_vertex, v, d_rewire, p_rewire))
              # Note: Propagation of cost change to children is omitted for simplicity in this basic RRT* implementation,
              # effectively making it RRT*-ish (optimal only at rewire step but consistent).

      if i % 100 == 0:
          c.draw_tree(vertex_list, edge_list, clear_draw_img=True)

          # Draw best path so far
          (closest_to_goal, d_goal) = vertex_list.get_closest_vertex_in_graph(goal_state)
          if d_goal < 0.1:
              path = []
              curr = closest_to_goal
              while curr:
                  path.append(curr.state)
                  curr = curr.parent
              c.draw_path(path)

          c.save_image_to_gif_image_stack()

  # Final output
  (closest_to_goal, d_goal) = vertex_list.get_closest_vertex_in_graph(goal_state)
  print(f"[Alpha {alpha}] Final Distance to Goal: {d_goal:.4f}")

  L = 0.0
  P = 0.0
  ratio = 0.0

  if d_goal < 0.1:
      L = closest_to_goal.dist_acc
      P = closest_to_goal.plastic_acc
      ratio = L / (P + 1e-6)
      print(f"Goal Reached! Total Distance: {L:.2f}, Total Plastic: {P:.2f}, Ratio (L/P): {ratio:.2f}")

      path = []
      curr = closest_to_goal
      while curr:
          path.append(curr.state)
          curr = curr.parent
      c.draw_path(path, color=(0, 255, 0), thickness_world=0.02, alpha=1.0)
  
  c.save_image_to_gif_image_stack()
  c.output_gif_animation(f'{output_prefix}_alpha_{alpha}_mode_{cost_mode}', fps=30, num_seconds_at_end=2)

  return L, P, ratio

if __name__ == '__main__':
    np.random.seed(42)
    # Run for alpha = 10.0 (Very High incentive for plastic)
    c3 = ContinuousPlannerUtil('ocean_real.png', 'summer_2002_day0_density.png', scale=0.8)
    rrt_star_tunable(c3, alpha=10.0, loops=800, cost_mode="linear")
