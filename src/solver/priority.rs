use crate::data::{Instance, OpId, Solution};
use ndarray::Array1;
use std::cmp;

pub fn find_solution_sps(inst: &Instance) -> Solution {
  return find_solution(&inst, &mut |candidates| {
    candidates
      .iter()
      .enumerate()
      .min_by_key(|&(_, op_id)| {
        let [j, o] = inst.op_from_id(*op_id);
        (o, j)
      })
      .unwrap()
      .0
  });
}

pub fn find_solution_lps(inst: &Instance) -> Solution {
  return find_solution(&inst, &mut |candidates| {
    candidates
      .iter()
      .enumerate()
      .max_by_key(|&(_, op_id)| {
        let [j, o] = inst.op_from_id(*op_id);
        (o, j)
      })
      .unwrap()
      .0
  });
}

pub fn find_solution_spt(inst: &Instance) -> Solution {
  return find_solution(&inst, &mut |candidates| {
    candidates
      .iter()
      .enumerate()
      .min_by_key(|&(_, &op_id)| {
        let [j, o] = inst.op_from_id(op_id);
        (inst.durations[op_id], j, o)
      })
      .unwrap()
      .0
  });
}

pub fn find_solution_lpt(inst: &Instance) -> Solution {
  return find_solution(&inst, &mut |candidates| {
    candidates
      .iter()
      .enumerate()
      .max_by_key(|&(_, &op_id)| {
        let [j, o] = inst.op_from_id(op_id);
        (inst.durations[op_id], j, o)
      })
      .unwrap()
      .0
  });
}

pub fn find_solution_lwrm(inst: &Instance) -> Solution {
  return find_solution(&inst, &mut |candidates| {
    candidates
      .iter()
      .enumerate()
      .min_by_key(|&(_, &op_id)| {
        let [j, o] = inst.op_from_id(op_id);
        (get_work_remaining(&inst, j, o), j, o)
      })
      .unwrap()
      .0
  });
}

pub fn find_solution_mwrm(inst: &Instance) -> Solution {
  return find_solution(&inst, &mut |candidates| {
    candidates
      .iter()
      .enumerate()
      .max_by_key(|&(_, &op_id)| {
        let [j, o] = inst.op_from_id(op_id);
        (get_work_remaining(&inst, j, o), j, o)
      })
      .unwrap()
      .0
  });
}

fn get_work_remaining(inst: &Instance, job: usize, op: usize) -> u32 {
  let mut work_remaining = 0;
  for upcoming_op in op..inst.n_machines {
    let upcoming_op_id = inst.op_to_id([job, upcoming_op]);
    work_remaining += inst.durations[upcoming_op_id];
  }
  return work_remaining;
}

// A Computational Study of Representations in Genetic Programming to Evolve Dispatching Rules for the Job Shop Scheduling Problem
pub fn find_solution(
  inst: &Instance,
  choose_next: &mut dyn FnMut(&Vec<OpId>) -> usize,
) -> Solution {
  let mut op_start_times = Array1::<u32>::from_elem(inst.n_ops(), 0);
  let mut machine_next_release = Array1::<u32>::from_elem(inst.n_machines, 0);
  let mut job_next_release = Array1::<u32>::from_elem(inst.n_jobs, 0);

  let mut ready = Vec::new();
  for j in 0..inst.n_jobs {
    ready.push(inst.op_to_id([j, 0]));
  }

  while !ready.is_empty() {
    // Finde the operation in the queue that finishes earliest
    let (earliest_completion_op, earliest_completion) = ready
      .iter()
      .map(|&op_id| {
        let [j, _o] = inst.op_from_id(op_id);
        let m = inst.machines[op_id];

        let release = cmp::max(job_next_release[j], machine_next_release[m]);
        let completion = release + inst.durations[op_id];

        (op_id, completion)
      })
      .min_by_key(|&(op_id, completion)| (completion, op_id))
      .unwrap();

    let target_machine = inst.machines[earliest_completion_op];

    // Find all operations on the same machine
    let ops_on_target_machine: Vec<usize> = ready
      .clone()
      .into_iter()
      .filter(|&op_id| inst.machines[op_id] == target_machine)
      .collect();

    // Only consider operations on the same machine that could be released before this one completes
    let candidates: Vec<usize> = ops_on_target_machine
      .into_iter()
      .filter(|&op_id| {
        let [j, _o] = inst.op_from_id(op_id);

        job_next_release[j] <= earliest_completion
      })
      .collect();

    let chosen_idx = choose_next(&candidates);
    let chosen_op = candidates[chosen_idx];

    let [j, o] = inst.op_from_id(chosen_op);
    let m = inst.machines[chosen_op];
    let release_time = cmp::max(job_next_release[j], machine_next_release[m]);
    let finish_time = release_time + inst.durations[chosen_op];
    // Update the release time tracking arrays
    op_start_times[chosen_op] = release_time;
    machine_next_release[m] = finish_time;
    job_next_release[j] = finish_time;

    // Remove from queue and push successor (if exists)
    ready.retain(|&op| op != chosen_op);
    if o < inst.n_machines - 1 {
      ready.push(inst.op_to_id([j, o + 1]));
    }
  }

  return Solution {
    start_times: op_start_times,
  };
}
