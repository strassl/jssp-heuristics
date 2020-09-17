pub mod hill_climber;
mod n1;
pub mod priority;
pub mod random_restart_hill_climber;
pub mod sequential;
pub mod simulated_annealing;
pub mod tabu_search;

use crate::data::{Edge, Instance, OpId, Solution};
use itertools::Itertools;
use log;
use ndarray::Array1;
use std::cmp;
use std::collections::VecDeque;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct IntermediateSolution {
  instance: Instance,

  oriented_conflict_edges: Vec<Edge>,

  // Derived from instance
  precedence_edges: Vec<Edge>,
  pre_job: Array1<Option<OpId>>,
  succ_job: Array1<Option<OpId>>,
  // Derived from orientation
  pre_machine: Array1<Option<OpId>>,
  succ_machine: Array1<Option<OpId>>,
  release_times: Array1<u32>,
  tail_times: Array1<u32>,
  path_times: Array1<u32>,
  cmax: u32,
}

impl IntermediateSolution {
  pub fn new(instance: Instance, oriented_conflict_edges: Vec<Edge>) -> Self {
    let precedence_edges = get_precedence_edges(&instance);
    let (pre_job, succ_job) = get_pre_succ_relations(&instance, &precedence_edges);
    let (pre_machine, succ_machine) = get_pre_succ_relations(&instance, &oriented_conflict_edges);

    let release_times = get_release_times_from_pre_succ_relations(
      &instance,
      &pre_job,
      &succ_job,
      &pre_machine,
      &succ_machine,
    );
    let tail_times = get_tail_times_from_pre_succ_relations(
      &instance,
      &pre_job,
      &succ_job,
      &pre_machine,
      &succ_machine,
    );
    let path_times = &release_times + &tail_times;
    let cmax = *path_times.iter().max().unwrap();

    Self {
      instance: instance,
      precedence_edges: precedence_edges,
      oriented_conflict_edges: oriented_conflict_edges,
      pre_job: pre_job,
      succ_job: succ_job,
      pre_machine: pre_machine,
      succ_machine: succ_machine,
      release_times: release_times,
      tail_times: tail_times,
      path_times: path_times,
      cmax: cmax,
    }
  }

  pub fn cmax(&self) -> u32 {
    return self.cmax;
  }

  pub fn to_solution(&self) -> Solution {
    return Solution {
      start_times: self.release_times.clone(),
    };
  }

  fn is_critical(&self, node: OpId) -> bool {
    return self.path_times[node] == self.cmax;
  }

  fn apply_swap(&self, a: OpId, b: OpId) -> IntermediateSolution {
    // Apply swap for neighborhood n1
    log::trace!("apply_swap({}, {})", a, b);
    log::trace!("oriented_conflict_edges={:?}", self.oriented_conflict_edges);
    let mut edges = self.oriented_conflict_edges.clone();

    // Reorient all affected edges
    for edge in &mut edges {
      let (u, v) = *edge;

      if u == a && v == b {
        *edge = (b, a);
      } else if v == a {
        *edge = (u, b);
      } else if u == b {
        *edge = (a, v);
      }
    }

    let new_oriented_conflict_edges = edges.into_iter().collect();
    log::trace!(
      "new_oriented_conflict_edges: {:?}",
      new_oriented_conflict_edges
    );

    let instance = self.instance.clone();
    let precedence_edges = self.precedence_edges.clone();
    let pre_job = self.pre_job.clone();
    let succ_job = self.succ_job.clone();

    let mut pre_machine = self.pre_machine.clone();
    let mut succ_machine = self.succ_machine.clone();

    // Apply edge swap to pre/succ relations
    if let Some(pre_machine_a) = self.pre_machine[a] {
      succ_machine[pre_machine_a] = Some(b);
    }
    if let Some(succ_machine_b) = self.succ_machine[b] {
      pre_machine[succ_machine_b] = Some(a);
    }
    pre_machine[a] = Some(b);
    succ_machine[a] = self.succ_machine[b];
    pre_machine[b] = self.pre_machine[a];
    succ_machine[b] = Some(a);

    let release_times = get_release_times_from_pre_succ_relations(
      &instance,
      &pre_job,
      &succ_job,
      &pre_machine,
      &succ_machine,
    );
    let tail_times = get_tail_times_from_pre_succ_relations(
      &instance,
      &pre_job,
      &succ_job,
      &pre_machine,
      &succ_machine,
    );

    let path_times = &release_times + &tail_times;
    let cmax = *path_times.iter().max().unwrap();

    Self {
      instance: instance,
      precedence_edges: precedence_edges,
      oriented_conflict_edges: new_oriented_conflict_edges,
      pre_job: pre_job,
      succ_job: succ_job,
      pre_machine: pre_machine,
      succ_machine: succ_machine,
      release_times: release_times,
      tail_times: tail_times,
      path_times: path_times,
      cmax: cmax,
    }
  }

  // Gives cmax if critical path passes through a or b but at least a lower bound on the new cmax
  fn cmax_after_swap(&self, a: OpId, b: OpId) -> u32 {
    let (a_new_release, a_new_tail, b_new_release, b_new_tail) = self.times_after_swap(a, b);

    let new_cmax = cmp::max(b_new_release + b_new_tail, a_new_release + a_new_tail);

    return new_cmax;
  }

  fn times_after_swap(&self, a: OpId, b: OpId) -> (u32, u32, u32, u32) {
    let pre_machine_a_end = if let Some(pre_machine_a) = self.pre_machine[a] {
      self.release_times[pre_machine_a] + self.instance.durations[pre_machine_a]
    } else {
      0
    };

    let pre_job_b_end = if let Some(pre_job_b) = self.pre_job[b] {
      self.release_times[pre_job_b] + self.instance.durations[pre_job_b]
    } else {
      0
    };

    let pre_job_a_end = if let Some(pre_job_a) = self.pre_job[a] {
      self.release_times[pre_job_a] + self.instance.durations[pre_job_a]
    } else {
      0
    };

    let succ_machine_b_tail = if let Some(succ_machine_b) = self.succ_machine[b] {
      self.tail_times[succ_machine_b]
    } else {
      0
    };

    let succ_job_a_tail = if let Some(succ_job_a) = self.succ_job[a] {
      self.tail_times[succ_job_a]
    } else {
      0
    };

    let succ_job_b_tail = if let Some(succ_job_b) = self.succ_job[b] {
      self.tail_times[succ_job_b]
    } else {
      0
    };

    let b_new_release = cmp::max(pre_machine_a_end, pre_job_b_end);
    let b_new_end = b_new_release + self.instance.durations[b];
    let a_new_release = cmp::max(b_new_end, pre_job_a_end);
    let a_new_tail = cmp::max(succ_machine_b_tail, succ_job_a_tail) + self.instance.durations[a];
    let b_new_tail = cmp::max(a_new_tail, succ_job_b_tail) + self.instance.durations[b];

    return (a_new_release, a_new_tail, b_new_release, b_new_tail);
  }
}

pub fn get_precedence_edges(inst: &Instance) -> Vec<Edge> {
  let mut edges = Vec::new();
  for j in 0..inst.n_jobs {
    for o in 1..inst.n_machines {
      let op = inst.op_to_id([j, o]);
      let pre_op = inst.op_to_id([j, o - 1]);
      edges.push((pre_op, op));
    }
  }

  return edges;
}

pub fn get_orientation_from_schedule(inst: &Instance, solution: &Solution) -> Vec<Edge> {
  let mut machine_to_operations = Array1::from_elem(inst.n_machines, Vec::new());
  for op in 0..inst.n_ops() {
    let m = inst.machines[op];
    machine_to_operations[m].push(op);
  }

  for m in 0..machine_to_operations.len() {
    let ops = &mut machine_to_operations[m];
    ops.sort_by(|&a, &b| {
      if is_before(a, b, &solution.start_times, &inst.durations) {
        cmp::Ordering::Less
      } else {
        cmp::Ordering::Greater
      }
    });
  }

  let mut edges = Vec::new();
  for m in 0..machine_to_operations.len() {
    let ops = &machine_to_operations[m];
    for (a, b) in ops.iter().tuple_windows() {
      edges.push((*a, *b));
    }
  }

  log::trace!("edges={:?}", edges);

  return edges;
}

pub fn is_before(a: OpId, b: OpId, release_times: &Array1<u32>, durations: &Array1<u32>) -> bool {
  let ord = op_ordering(a, b, release_times, durations);

  return match ord {
    std::cmp::Ordering::Less => true,
    std::cmp::Ordering::Greater => false,
    std::cmp::Ordering::Equal => {
      panic!(
        "Overlapping jobs {}:[{}+{}]; {}:[{}+{}]",
        a, release_times[a], durations[a], b, release_times[b], durations[b]
      );
    }
  };
}

pub fn op_ordering(
  a: OpId,
  b: OpId,
  release_times: &Array1<u32>,
  durations: &Array1<u32>,
) -> cmp::Ordering {
  let r_a = release_times[a];
  let r_b = release_times[b];

  if r_a < r_b {
    return cmp::Ordering::Less;
  } else if r_a == r_b {
    // If at least one job is zero-length it should still be before the other job, even if the release times are the same
    let d_a = durations[a];
    let d_b = durations[b];

    if d_a == 0 && d_b != 0 {
      return cmp::Ordering::Less;
    } else if d_a != 0 && d_b == 0 {
      return cmp::Ordering::Greater;
    } else if d_a == 0 && d_b == 0 {
      // Fall back to ordering by op id
      return a.cmp(&b);
    } else {
      return cmp::Ordering::Equal;
    }
  } else {
    return cmp::Ordering::Greater;
  }
}

fn get_release_times_from_pre_succ_relations(
  inst: &Instance,
  pre_job: &Array1<Option<OpId>>,
  succ_job: &Array1<Option<OpId>>,
  pre_machine: &Array1<Option<OpId>>,
  succ_machine: &Array1<Option<OpId>>,
) -> Array1<u32> {
  let mut release_time = Array1::<Option<u32>>::from_elem(inst.n_ops(), None);
  let mut labelled = Array1::<bool>::from_elem(inst.n_ops(), false);

  let mut open = VecDeque::new();
  for op in 0..inst.n_ops() {
    if pre_job[op] == None && pre_machine[op] == None {
      open.push_back(op);
      release_time[op] = Some(0);
    }
  }

  while !open.is_empty() {
    let node = open.pop_front().unwrap();

    let pre_job_end = if let Some(pre_job_node) = pre_job[node] {
      release_time[pre_job_node].unwrap() + inst.durations[pre_job_node]
    } else {
      0
    };

    let pre_machine_end = if let Some(pre_machine_node) = pre_machine[node] {
      release_time[pre_machine_node].unwrap() + inst.durations[pre_machine_node]
    } else {
      0
    };

    let release = cmp::max(pre_job_end, pre_machine_end);
    release_time[node] = Some(release);
    labelled[node] = true;

    if let Some(succ_job_node) = succ_job[node] {
      if let Some(pre_machine_succ_job_node) = pre_machine[succ_job_node] {
        if labelled[pre_machine_succ_job_node] {
          open.push_back(succ_job_node);
        }
      } else {
        open.push_back(succ_job_node);
      }
    }

    if let Some(succ_machine_node) = succ_machine[node] {
      if let Some(pre_job_succ_machine_node) = pre_job[succ_machine_node] {
        if labelled[pre_job_succ_machine_node] {
          open.push_back(succ_machine_node);
        }
      } else {
        open.push_back(succ_machine_node);
      }
    }
  }

  if log::log_enabled!(log::Level::Trace) {
    if release_time.iter().any(|x| x.is_none()) {
      let op_rels = inst
        .op_ids()
        .iter()
        .map(|&op| {
          let default = "_".to_string();
          format!(
            "[{} -> {} | pm={}, pj={}, sm={}, sj={}]",
            op,
            &release_time[op]
              .map(|x| x.to_string())
              .unwrap_or(default.clone()),
            &pre_machine[op]
              .map(|x| x.to_string())
              .unwrap_or(default.clone()),
            &pre_job[op]
              .map(|x| x.to_string())
              .unwrap_or(default.clone()),
            &succ_machine[op]
              .map(|x| x.to_string())
              .unwrap_or(default.clone()),
            &succ_job[op]
              .map(|x| x.to_string())
              .unwrap_or(default.clone()),
          )
        })
        .collect::<Vec<_>>()
        .join(", ");
      log::trace!("op_rels=[{}]", op_rels);
    }
  }

  return release_time.map(|r| r.unwrap());
}

// see Taillard, Parallel Taboo Search Techniques for the Job Shop Scheduling Problem
fn get_tail_times_from_pre_succ_relations(
  inst: &Instance,
  pre_job: &Array1<Option<OpId>>,
  succ_job: &Array1<Option<OpId>>,
  pre_machine: &Array1<Option<OpId>>,
  succ_machine: &Array1<Option<OpId>>,
) -> Array1<u32> {
  let mut open = VecDeque::new();
  for op in 0..inst.n_ops() {
    if succ_job[op] == None && succ_machine[op] == None {
      open.push_back(op);
    }
  }

  let mut labelled = Array1::<bool>::from_elem(inst.n_ops(), false);
  let mut tail_time = Array1::<u32>::from_elem(inst.n_ops(), 0);
  while !open.is_empty() {
    let node = open.pop_front().unwrap();

    let succ_job_tail = if let Some(succ_job_node) = succ_job[node] {
      tail_time[succ_job_node]
    } else {
      0
    };

    let succ_machine_tail = if let Some(succ_machine_node) = succ_machine[node] {
      tail_time[succ_machine_node]
    } else {
      0
    };

    let tail = cmp::max(succ_job_tail, succ_machine_tail) + inst.durations[node];
    tail_time[node] = tail;
    labelled[node] = true;

    if let Some(pre_job_node) = pre_job[node] {
      if let Some(succ_machine_pre_job_node) = succ_machine[pre_job_node] {
        if labelled[succ_machine_pre_job_node] {
          open.push_back(pre_job_node);
        }
      } else {
        open.push_back(pre_job_node);
      }
    }

    if let Some(pre_machine_node) = pre_machine[node] {
      if let Some(succ_job_pre_machine_node) = succ_job[pre_machine_node] {
        if labelled[succ_job_pre_machine_node] {
          open.push_back(pre_machine_node);
        }
      } else {
        open.push_back(pre_machine_node);
      }
    }
  }

  return tail_time;
}

pub fn get_pre_succ_relations(
  inst: &Instance,
  edges: &Vec<Edge>,
) -> (Array1<Option<OpId>>, Array1<Option<OpId>>) {
  let mut pre = Array1::<Option<OpId>>::from_elem(inst.n_ops(), None);
  let mut succ = Array1::<Option<OpId>>::from_elem(inst.n_ops(), None);

  for &(v, w) in edges {
    if let Some(u) = pre[w] {
      panic!("Duplicate when inserting {} pre[{}]={}", v, w, u);
    }
    if let Some(u) = succ[v] {
      panic!("Duplicate when inserting {} succ[{}]={}", w, v, u);
    }
    pre[w] = Some(v);
    succ[v] = Some(w);
  }

  return (pre, succ);
}

pub fn verify_solution(inst: &Instance, solution: &Solution) -> Result<(), Box<dyn Error>> {
  // Check:
  // 1. For every job: order
  // 2. For every job: no overlap
  // 3. For every machine: no overlap

  for job in 0..inst.n_jobs {
    for op in 0..inst.n_machines {
      let op_id = inst.op_to_id([job, op]);
      let machine = inst.machines[op_id];
      let duration = inst.durations[op_id];
      let start = solution.start_times[op_id];
      let end = start + duration;

      // Who needs decent runtime complexity any way
      for other_job in 0..inst.n_jobs {
        for other_op in 0..inst.n_machines {
          if other_job != job && other_op != op {
            let other_op_id = inst.op_to_id([other_job, other_op]);
            let other_machine = inst.machines[other_op_id];
            let other_duration = inst.durations[other_op_id];
            let other_start = solution.start_times[other_op_id];
            let other_end = other_start + other_duration;

            if other_job == job && other_op + 1 == op {
              if other_end > start {
                Err(format!(
                  "Precedence violation in job {:?} - {:?}:[{:?}, {:?}] should be before {:?}:[{:?}, {:?}]",
                  job,
                  [job, op],
                  start,
                  end,
                  [other_job, other_op],
                  other_start,
                  other_end
                ))?;
              }
            }

            if other_machine == machine {
              if !((start <= other_start && end <= other_start)
                || (other_start <= start && other_end <= start))
              {
                Err(format!(
                  "Overlap in machine {:?} - {:?}:[{:?}, {:?}] overlaps with {:?}:[{:?}, {:?}]",
                  machine,
                  [job, op],
                  start,
                  end,
                  [other_job, other_op],
                  other_start,
                  other_end
                ))?;
              }
            }
          }
        }
      }
    }
  }

  Ok(())
}

pub fn print_solution(inst: &Instance, solution: &Solution) {
  for job in 0..inst.n_jobs {
    let mut starts = Vec::new();
    for op in 0..inst.n_machines {
      let start = solution.start_times[inst.op_to_id([job, op])];
      starts.push(start.to_string());
    }
    let line = starts.join(" ");
    println!("{}", line);
  }
}

pub fn calculate_cmax(inst: &Instance, solution: &Solution) -> u32 {
  return calculate_cmax_from_release_times(&inst, &solution.start_times);
}

fn calculate_cmax_from_release_times(inst: &Instance, release_times: &Array1<u32>) -> u32 {
  let mut cmax = 0;

  for op in 0..inst.n_ops() {
    let duration = inst.durations[op];
    let start = release_times[op];
    let end = start + duration;

    cmax = cmp::max(cmax, end);
  }

  return cmax;
}

pub fn generate_random_solution<R: rand::Rng>(inst: &Instance, rng: &mut R) -> Solution {
  let mut op_start_times = Array1::<u32>::from_elem(inst.n_ops(), 0);
  let mut machine_next_release = Array1::<u32>::from_elem(inst.n_machines, 0);
  let mut job_next_release = Array1::<u32>::from_elem(inst.n_jobs, 0);

  let mut ready = Vec::new();
  for j in 0..inst.n_jobs {
    ready.push(inst.op_to_id([j, 0]));
  }

  while !ready.is_empty() {
    let chosen_idx = rng.gen_range(0, ready.len());
    let chosen_op = ready.remove(chosen_idx);

    let [j, o] = inst.op_from_id(chosen_op);
    let m = inst.machines[chosen_op];
    let release_time = cmp::max(job_next_release[j], machine_next_release[m]);
    let finish_time = release_time + inst.durations[chosen_op];
    // Update the release time tracking arrays
    op_start_times[chosen_op] = release_time;
    machine_next_release[m] = finish_time;
    job_next_release[j] = finish_time;

    if o < inst.n_machines - 1 {
      ready.push(inst.op_to_id([j, o + 1]));
    }
  }

  return Solution {
    start_times: op_start_times,
  };
}
