use crate::data::{Instance, Solution};
use ndarray::Array1;

pub fn find_solution(inst: &Instance) -> Solution {
  let mut op_start_times = Array1::<u32>::from_elem(inst.n_ops(), 0);

  let mut next_start_time = 0;
  for j in 0..inst.n_jobs {
    for o in 0..inst.n_machines {
      let op = [j, o];
      let op_id = inst.op_to_id(op);
      let start = next_start_time;
      let end = start + inst.durations[op_id];

      op_start_times[op_id] = start;
      next_start_time = end;
    }
  }

  return Solution {
    start_times: op_start_times,
  };
}
